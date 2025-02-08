# Copyright 2024 Lumen Lab AI Work Group, chat REPL for OpenGrok
# 
# This script loads a pre-trained causal language model from the specified model path
# and uses it to generate responses based on a system prompt and user input. The model 
# and tokenizer are loaded from the OpenGrok model directory, and the system prompt is 
# read from 'system_prompt.md'. The script handles tokenization, model inference, and 
# generates a response based on the input prompt using the model. 
# 
# The program handles errors related to missing files and other exceptions gracefully, 
# with appropriate messages displayed for common issues such as missing system prompt file.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = './models/OpenGrok'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    # 处理可能的pad_token缺失问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('system_prompt.md', 'r') as raw_system_prompt:
        system_prompt = raw_system_prompt.read()

    # 自动设备映射与量化配置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    
    prompt = input("Enter your prompt: ")
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 设备感知的输入处理
    model_inputs = tokenizer(text, return_tensors='pt').to(model.device)
  
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,  # 避免警告
            eos_token_id=tokenizer.eos_token_id,  # 使用标准eos_token
            do_sample=True,  # 可选：启用采样策略
            temperature=0.7,  # 可选：控制随机性
            top_p=0.9  # 可选：核采样
        )

    input_length = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0, input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # 清理可能的分隔符
    response = response.split('<|endoftext|>')[0]
    response = response.split('<|end|>')[0]
    
    print(response)

except FileNotFoundError:
    print("Error: system_prompt.md file not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
