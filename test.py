import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def analyze_processor(image_path, question, repeat_k=None):
    """分析处理器如何处理输入"""
    
    # 1. 加载模型和处理器
    print("\n1. 加载模型和处理器...")
    model = LlavaForConditionalGeneration.from_pretrained(
        "../autodl-tmp/model/llava-1.5-7b-hf/swift/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    
    processor = AutoProcessor.from_pretrained("../autodl-tmp/model/llava-1.5-7b-hf/swift/llava-1.5-7b-hf")
    processor.patch_size = model.config.vision_config.patch_size
    
    # 2. 加载和显示图像信息
    print("\n2. 加载图像...")
    image = Image.open(image_path).convert('RGB')
    print(f"原始图像大小: {image.size}")
    print(f"图像模式: {image.mode}")
    
    # 3. 构造对话模板
    print("\n3. 构造对话模板...")
    processor.chat_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {image_tokens} {question} ASSISTANT:"""

    # USER: <ImageHere> <question> ASSISTANT:
    
    # 4. 分析Vanilla策略
    print("\n4. 分析Vanilla策略...")
    # 4.1 构造对话历史
    vanilla_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    
    # 4.2 生成prompt
    vanilla_prompt = processor.apply_chat_template(
        vanilla_conversation,
        add_generation_prompt=True,
        chat_template=processor.chat_template.format(
            image_tokens="<image>",
            question=question
        )
    )
    print("\nVanilla Prompt:")
    print(vanilla_prompt)
    
    # 4.3 处理输入
    vanilla_inputs = processor(
        images=[image],
        text=vanilla_prompt,
        return_tensors='pt'
    )
    
    print("\nVanilla Inputs:")
    for key, value in vanilla_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")
    
    # 5. 如果指定了repeat_k，分析Repeat策略
    if repeat_k:
        print(f"\n5. 分析Repeat策略 (k={repeat_k})...")
        # 5.1 构造重复的对话历史
        repeat_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in range(repeat_k)
                ] + [{"type": "text", "text": question}],
            },
        ]
        
        # 5.2 生成prompt
        # image_tokens = " ".join(["<image>"] * repeat_k)
        image_tokens = "<image> <image>"
        repeat_prompt = processor.apply_chat_template(
            repeat_conversation,
            add_generation_prompt=True,
            chat_template=processor.chat_template.format(
                image_tokens=image_tokens,
                question=question
            )
        )
        print("\nRepeat Prompt:")
        print(repeat_prompt)
        
        # 5.3 处理输入
        repeat_inputs = processor(
            images=[image] * repeat_k,
            text=repeat_prompt,
            return_tensors='pt'
        )
        
        print("\nRepeat Inputs:")
        for key, value in repeat_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")
        
        # 5.4 比较两种策略的区别
        print("\n6. 比较两种策略的区别:")
        for key in vanilla_inputs.keys():
            if isinstance(vanilla_inputs[key], torch.Tensor):
                print(f"\n{key}:")
                print(f"Vanilla shape: {vanilla_inputs[key].shape}")
                print(f"Repeat shape: {repeat_inputs[key].shape}")
                if key == "pixel_values":
                    print(f"图像特征是否相同: {torch.allclose(repeat_inputs[key][0], repeat_inputs[key][1])}")

if __name__ == "__main__":
    # 使用一张示例图片
    image_path = "./teaser.png"  # 替换为实际的图片路径
    question = "Is there a people in the image?"
    analyze_processor(image_path, question, repeat_k=2) 