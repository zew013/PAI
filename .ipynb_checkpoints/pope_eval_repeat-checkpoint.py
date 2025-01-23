"""
运行前请确保安装以下依赖:
pip install torch transformers pillow tqdm accelerate einops timm peft
pip install llmtuner  # LLaMA Factory

如果使用的是LLaVA模型，还需要安装:
pip install git+https://github.com/haotian-liu/LLaVA.git@v1.1.3
"""

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import process_images
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from collections import defaultdict
import hashlib
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token

# 设置随机种子
def setup_seeds(seed=927):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# 添加采样函数
def sample_pope_subset(samples: List[dict], yes_ratio: float, total_samples: int, seed: int = 927) -> List[dict]:
    """
    从POPE数据集中采样子集
    
    Args:
        samples: 所有样本列表
        yes_ratio: yes样本的目标比例
        total_samples: 需要的总样本数
        seed: 随机种子
    
    Returns:
        采样的子集列表
    """
    # 设置随机种子
    random.seed(seed)
    
    # 分离yes和no样本
    yes_samples = [s for s in samples if s['label'] == 'yes']
    no_samples = [s for s in samples if s['label'] == 'no']
    
    # 计算需要的yes和no样本数
    yes_count = int(total_samples * yes_ratio)
    no_count = total_samples - yes_count
    
    # 确保有足够的样本
    if len(yes_samples) < yes_count or len(no_samples) < no_count:
        raise ValueError(
            f"没有足够的样本。需要 {yes_count} yes样本和 {no_count} no样本，"
            f"但只有 {len(yes_samples)} yes样本和 {len(no_samples)} no样本。"
        )
    
    # 随机采样
    sampled_yes = random.sample(yes_samples, yes_count)
    sampled_no = random.sample(no_samples, no_count)
    
    # 合并并打乱
    sampled_subset = sampled_yes + sampled_no
    random.shuffle(sampled_subset)
    
    return sampled_subset

def get_subset_path(pope_path: str, yes_ratio: float, total_samples: int) -> str:
    """生成子集文件路径"""
    # 使用原始文件名、yes比例和样本总数生成唯一的子集文件名
    base_name = os.path.splitext(os.path.basename(pope_path))[0]
    subset_name = f"{base_name}_yes{yes_ratio:.2f}_n{total_samples}.json"
    return os.path.join(os.path.dirname(pope_path), "subsets", subset_name)

# POPE数据集加载器
class POPEDataset(Dataset):
    def __init__(self, 
                 metadata_path: str,
                 image_dir: str, 
                 image_processor=None,
                 yes_ratio: float = None,
                 total_samples: int = -1):
        self.image_dir = image_dir
        self.image_processor = image_processor
        
        # 加载metadata
        with open(metadata_path, 'r') as f:
            all_samples = json.load(f)
            
        # 将answer转换为标准格式
        for sample in all_samples:
            sample['label'] = 'yes' if sample['answer'].lower() == 'yes' else 'no'
            sample['text'] = sample['question']
            sample['image'] = f"{sample['image_source']}.jpg"
        
        # 如果指定了采样参数且total_samples不为-1，创建或加载子集
        if yes_ratio is not None and total_samples > 0:
            subset_dir = os.path.join(os.path.dirname(metadata_path), "subsets")
            os.makedirs(subset_dir, exist_ok=True)
            
            subset_path = get_subset_path(metadata_path, yes_ratio, total_samples)
            
            if os.path.exists(subset_path):
                print(f"加载已存在的子集: {subset_path}")
                with open(subset_path, 'r') as f:
                    subset_data = json.load(f)
                    self.samples = subset_data['samples']
                    self.metadata = subset_data['metadata']
            else:
                print(f"创建新的子集: {subset_path}")
                sampled_subset = sample_pope_subset(all_samples, yes_ratio, total_samples)
                
                # 记录子集信息
                self.metadata = {
                    'original_path': metadata_path,
                    'yes_ratio': yes_ratio,
                    'total_samples': total_samples,
                    'actual_yes_ratio': sum(1 for x in sampled_subset if x['label'] == 'yes') / len(sampled_subset),
                    'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'categories': self._get_category_distribution(sampled_subset)
                }
                
                # 保存子集
                with open(subset_path, 'w') as f:
                    json.dump({
                        'metadata': self.metadata,
                        'samples': sampled_subset
                    }, f, indent=2)
                
                self.samples = sampled_subset
        else:
            # 使用完整数据集
            self.samples = all_samples
            self.metadata = {
                'original_path': metadata_path,
                'total_samples': len(self.samples),
                'yes_ratio': sum(1 for x in self.samples if x['label'] == 'yes') / len(self.samples),
                'categories': self._get_category_distribution(self.samples)
            }
        
        print(f"\n数据集信息:")
        print(f"总样本数: {len(self.samples)}")
        print(f"Yes样本比例: {self.metadata['yes_ratio']:.2%}")
        print("\n类别分布:")
        for cat, count in self.metadata['categories'].items():
            print(f"{cat}: {count}")

    def _get_category_distribution(self, samples):
        """统计类别分布"""
        category_counts = defaultdict(int)
        for sample in samples:
            category_counts[sample['category']] += 1
        return dict(category_counts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"图片目录内容: {os.listdir(self.image_dir)}")
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 不在这里处理图像，直接返回原始图像
            return {
                "image": image,
                "text": item["text"],
                "label": 1 if item["label"] == "yes" else 0
            }
                
        except Exception as e:
            print(f"图片加载失败: {str(e)}")
            print(f"图片路径: {image_path}")
            raise

class PromptStrategy:
    def __init__(self, processor, verbose=False):
        self.processor = processor
        self.verbose = verbose
        # 设置对话模板
#         self.processor.chat_template = """
# A chat between a curious user and an AI assistant. The assistant provides helpful, detailed, and polite answers to the user's questions.
# USER: {image_tokens}
# {question}
# ASSISTANT:
# """
        self.processor.chat_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {image_tokens}
{question}
ASSISTANT:"""

    def build_vanilla_prompt(self, question: str) -> str:
        # 构建对话历史
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        # 使用单个图像标记
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            chat_template=self.processor.chat_template.format(
                image_tokens="<image>",
                question=question
            )
        )
        if self.verbose:
            print("\nVanilla Prompt:")
            print(prompt)
        return prompt
    
    def build_repeat_prompt(self, question: str, k: int = 2) -> str:
        # 构建带有重复图像的对话历史
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in range(k)
                ] + [{"type": "text", "text": question}],
            },
        ]

        # 构建重复的图像标记
        image_tokens = " ".join(["<image>"] * k)

        # 使用重复的图像标记
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            chat_template=self.processor.chat_template.format(
                image_tokens=image_tokens,
                question=question
            )
        )

        if self.verbose:
            print(f"\nRepeat Prompt (k={k}):")
            print(prompt)
            print("\nImage tokens:", image_tokens)

        # 在处理输入时也打印token信息
        test_tokens = self.processor.tokenizer(prompt, return_tensors='pt')
        if self.verbose:
            print("\nTokenized length:", test_tokens.input_ids.shape)
            print("Special tokens:", [
                self.processor.tokenizer.decode([token_id]) 
                for token_id in test_tokens.input_ids[0] 
                if token_id in [self.processor.tokenizer.pad_token_id, 
                              self.processor.tokenizer.eos_token_id,
                              self.processor.tokenizer.bos_token_id]
            ])

        return prompt

def evaluate_pope(predictions, labels):
    """评估POPE结果，计算accuracy、precision、recall和f1
    
    Args:
        predictions: 模型预测的文本列表
        labels: 真实标签列表(1表示yes, 0表示no)
    
    Returns:
        包含各项指标的字典
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    invalid_count = 0
    
    for pred, label in zip(predictions, labels):
        pred_lower = pred.lower()
        
        # 检查是否包含明确的yes/no词
        neg_words = ["no", "not", "cannot", "n't", "neither", "nor"]
        pos_words = ["yes", "yeah", "yep", "correct", "right"]
        
        is_negative = any(word in pred_lower for word in neg_words) or \
                     any(word.endswith("n't") for word in pred_lower.split())
        is_positive = any(word in pred_lower for word in pos_words)
        
        if is_negative and is_positive:
            # 如果同时包含肯定和否定词，记为invalid
            invalid_count += 1
            continue
        elif is_negative:
            pred_label = 0
        elif is_positive:
            pred_label = 1
        else:
            # 既没有肯定也没有否定词，记为invalid
            invalid_count += 1
            continue
        
        # 统计混淆矩阵
        if pred_label == 1 and label == 1:
            TP += 1
        elif pred_label == 1 and label == 0:
            FP += 1
        elif pred_label == 0 and label == 0:
            TN += 1
        else:  # pred_label == 0 and label == 1
            FN += 1
    
    total_valid = TP + TN + FP + FN
    total = len(predictions)
    
    # 计算指标
    accuracy = (TP + TN) / total if total_valid > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算各类答案的比例
    yes_ratio = sum(1 for pred in predictions if any(word in pred.lower() for word in pos_words)) / total
    no_ratio = sum(1 for pred in predictions if any(word in pred.lower() for word in neg_words)) / total
    invalid_ratio = invalid_count / total
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "no_ratio": no_ratio,
        "invalid_ratio": invalid_ratio,
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },
        "counts": {
            "total": total,
            "valid": total_valid,
            "invalid": invalid_count,
            "yes": int(yes_ratio * total),
            "no": int(no_ratio * total)
        }
    }

# 添加自定义的collate函数
def custom_collate_fn(batch):
    """自定义collate函数，用于处理包含PIL Image的batch"""
    elem = batch[0]
    batch_dict = {
        "image": [d["image"] for d in batch],  # 保持PIL Image格式
        "text": [d["text"] for d in batch],
        "label": torch.tensor([d["label"] for d in batch])
    }
    return batch_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pope-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, choices=['vanilla', 'repeat'], default='vanilla',
                      help="选择prompt策略: vanilla或repeat")
    parser.add_argument("--repeat-k", type=int, default=1,
                      help="repeat策略时的重复次数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--yes-ratio", type=float, help="子集中yes样本的比例")
    parser.add_argument("--total-samples", type=int, default=-1,
                      help="子集总样本数，-1表示使用全部数据")
    parser.add_argument("--offload-folder", type=str, default="./offload",
                      help="模型权重卸载目录")
    parser.add_argument("--max-memory", type=dict, default=None,
                      help="每个GPU设备的最大内存使用量")
    parser.add_argument("--num-gpus", type=int, default=1,
                      help="使用的GPU数量")
    parser.add_argument("--verbose", action="store_true", default=False,
                      help="是否输出详细信息")
    args = parser.parse_args()

    # 添加打印函数
    def verbose_print(*print_args, **kwargs):
        if args.verbose:
            print(*print_args, **kwargs)

    # 设置随机种子
    setup_seeds()
    
    # 禁用torch初始化
    disable_torch_init()

    try:
        print(f"正在加载模型: {args.model_path}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        # 加载模型和处理器
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="balanced"
        ).eval()
        
        processor = AutoProcessor.from_pretrained(args.model_path)
        # 设置patch_size
        processor.patch_size = model.config.vision_config.patch_size
        
        verbose_print("模型加载成功")
        verbose_print(f"模型分布: {model.hf_device_map}")
        verbose_print(f"Vision patch size: {processor.patch_size}")
        
        # 加载数据集
        dataset = POPEDataset(
            args.pope_path, 
            args.image_dir,
            None,  # 不需要image_processor
            yes_ratio=args.yes_ratio,
            total_samples=args.total_samples
        )
        
        # 使用DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # 初始化prompt策略 - 使用processor
        prompt_strategy = PromptStrategy(processor, verbose=args.verbose)
        
        # 评估结果存储
        results = {
            "predictions": [],
            "labels": []
        }
        
        # 开始评估
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    current_gpu = batch_idx % min(args.num_gpus, torch.cuda.device_count())
                    
                    for i in range(len(batch["image"])):
                        try:
                            image = batch["image"][i]  # PIL Image
                            question = batch["text"][i]
                            label = batch["label"][i]
                            
                            # 构建prompt
                            if args.strategy == "vanilla":
                                prompt = prompt_strategy.build_vanilla_prompt(question)
                                images = [image]
                            else:  # repeat
                                prompt = prompt_strategy.build_repeat_prompt(question, args.repeat_k)
                                images = [image] * args.repeat_k 

                            # 打印调试信息
                            verbose_print(f"\n处理第 {batch_idx * args.batch_size + i + 1} 个样本 (GPU {current_gpu})")
                            verbose_print(f"Prompt: {prompt}")
                            verbose_print(f"传递的图像数量: {len(images)}")
                            image_token_count = prompt.count("<image>")
                            verbose_print(f"Prompt中 <image> 的数量: {image_token_count}")
                            if len(images) != image_token_count:
                                print(f"[Error] 图像数量 ({len(images)}) 与 <image> 令牌数量 ({image_token_count}) 不匹配！")

                            # 在指定GPU上运行
                            with torch.cuda.device(current_gpu):
                                try:
                                    # 处理输入
                                    inputs = processor(
                                        images=images,
                                        text=prompt,
                                        return_tensors='pt'
                                    ).to(current_gpu, torch.float16)
                                    
                                    # 生成回答
                                    output_ids = model.generate(
                                        **inputs,
                                        use_cache=True,
                                        do_sample=False,
                                        max_new_tokens=512,
                                        min_new_tokens=1,
                                        num_beams=1,
                                        # pad_token_id=processor.tokenizer.pad_token_id,
                                        # eos_token_id=processor.tokenizer.eos_token_id,
                                    )
                                    
                                    # 解码输出
                                    output_text = processor.decode(
                                        output_ids[0][2:],  # 跳过前两个token
                                        skip_special_tokens=True
                                    ).strip()
                                    
                                    # 提取ASSISTANT之后的回答
                                    if "ASSISTANT:" in output_text:
                                        output_text = output_text.split("ASSISTANT:")[-1].strip()

                                    verbose_print(f"生成的回答: {output_text}")
                                    
                                    # 存储结果
                                    results["predictions"].append(output_text)
                                    results["labels"].append(label.item())
                                    
                                    # 清理当前GPU缓存
                                    torch.cuda.empty_cache()
                                
                                except torch.cuda.OutOfMemoryError as e:
                                    verbose_print(f"GPU {current_gpu} 内存不足: {str(e)}")
                                    torch.cuda.empty_cache()
                                    continue
                                except Exception as e:
                                    verbose_print(f"处理输入时发生错误: {str(e)}")
                                    verbose_print(f"Prompt: {prompt}")
                                    if 'input_ids' in locals() and inputs.get('input_ids') is not None:
                                        verbose_print(f"Inputs shape: {inputs['input_ids'].shape}")
                                    else:
                                        verbose_print(f"Inputs: {inputs.keys() if inputs is not None else 'None'}")
                                    verbose_print(f"Image size: {image.size}")
                                    verbose_print(f"Image mode: {image.mode}")
                                    continue

                        except Exception as e:
                            verbose_print(f"处理样本时发生错误: {str(e)}")
                            verbose_print(f"错误类型: {type(e)}")
                            verbose_print(f"错误详情: {str(e)}")
                            verbose_print(f"图片信息: {image.size if isinstance(image, Image.Image) else 'Not a PIL Image'}")
                            verbose_print(f"图片模式: {image.mode if isinstance(image, Image.Image) else 'Unknown'}")
                            continue
                            
        finally:
            # 清理所有GPU资源
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            del model
            del processor
            
        # 评估结果
        metrics = evaluate_pope(
            results["predictions"],
            results["labels"]
        )
        
        # 始终打印结果统计
        print(f"\n{args.strategy.capitalize()} Strategy Results:")
        print(f"Total Samples: {metrics['counts']['total']}")
        print(f"Valid Samples: {metrics['counts']['valid']}")
        print(f"Invalid Samples: {metrics['counts']['invalid']}")
        print(f"Yes Responses: {metrics['counts']['yes']}")
        print(f"No Responses: {metrics['counts']['no']}")
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"\nResponse Ratios:")
        print(f"Yes Ratio: {metrics['yes_ratio']:.4f}")
        print(f"No Ratio: {metrics['no_ratio']:.4f}")
        print(f"Invalid Ratio: {metrics['invalid_ratio']:.4f}")
        print("\nConfusion Matrix:")
        print(f"TP: {metrics['confusion_matrix']['TP']}, FP: {metrics['confusion_matrix']['FP']}")
        print(f"FN: {metrics['confusion_matrix']['FN']}, TN: {metrics['confusion_matrix']['TN']}")
        
        # 获取当前时间戳
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 构建输出文件名
        output_file = f"pope_eval_{args.strategy}_k{args.repeat_k}_{timestamp}.json"
        
        # 创建输出目录
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        
        # 保存详细结果
        with open(output_path, "w") as f:
            json.dump({
                "dataset_info": dataset.metadata,
                "strategy": args.strategy,
                "repeat_k": args.repeat_k if args.strategy == "repeat" else None,
                "metrics": metrics,
                "predictions": results["predictions"],
                "labels": results["labels"],
                "timestamp": timestamp,  # 也在结果中保存时间戳
                "args": vars(args)  # 保存所有运行参数
            }, f, indent=2)
            
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("\nGPU内存不足，建议尝试:")
            print("1. 减小batch_size")
            print("2. 使用更小的模型")
            print("3. 增加--max-memory参数")
        raise
    
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 删除临时文件
        if os.path.exists(args.offload_folder):
            import shutil
            shutil.rmtree(args.offload_folder)

if __name__ == "__main__":
    main() 
    
    
# python pope_eval_repeat.py \
#     --model-path './autodl-tmp/model/llava-1.5-7b-hf' \
#     --pope-path ./autodl_tmp/pope_metadata.json \
#     --image-dir ./autodl_tmp/pope_images \
#     --strategy vanilla