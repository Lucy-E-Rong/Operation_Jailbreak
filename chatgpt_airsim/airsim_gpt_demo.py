import argparse
import re
import os
import math
import datetime
import json
from openai import OpenAI

# 初始化 OpenAI 客户端
client = None

def initialize_openai(api_key):
    global client
    client = OpenAI(api_key=api_key)
def ensure_client_initialized():
    global client
    if client is None:
        # 从配置文件或环境变量加载API密钥
        config_path = "chatgpt_airsim/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            api_key = config.get("OPENAI_API_KEY")
            if api_key:
                initialize_openai(api_key)
                return
        raise ValueError("OpenAI client not initialized and no API key found")
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
print(parent_dir)

with open("chatgpt_airsim/prompts/interpretation.txt", "r") as f:
    interpret_sysprompt = f.read()
with open("chatgpt_airsim/prompts/verification.txt", "r") as f:
    guard_sysprompt = f.read()
with open("chatgpt_airsim/system_prompts/airsim_basic.txt", "r") as f:
    sysprompt = f.read()

interpret_history = [
    {
        "role": "system",
        "content": interpret_sysprompt
    }
]
guard_history = [
    {
        "role": "system",
        "content": guard_sysprompt
    }
]

model_name = "gpt-4o"  # 默认模型"gpt-4o"o4-mini

def request_single_step(prompt):  #基础代码生成函数
    ensure_client_initialized()
    chat = [
        {
            "role": "system",
            "content": sysprompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=chat
    )
    response = completion.choices[0].message.content
    return response

def single_request(history, prompt, model_name):   #用于evaluation
    single_chat = history.copy()
    single_chat.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    completion = client.chat.completions.create(
        model = model_name, # "gpt-4", "gpt-4o-mini", "gpt-4-turbo"
        messages = single_chat
    )
    response = completion.choices[0].message.content

    return response
    
def evaluation(code):
    chat_interpret = interpret_history.copy()
    chat_verify = guard_history.copy()
    # refine loop
    interpretation = single_request(chat_interpret, code, model_name)
    verification = single_request(chat_verify, interpretation, model_name)

    if "VIOLATION:" in verification:
        response = f'Request denied. The user request violates: {verification}'
        return response

    elif "ALL CONSTRAINTS SATISFIED" in verification:
        response = code
        return response



'''
def main():
    print(f"Guard using {model_name}")


    # chat_code = chat_history.copy()
    for code in [code1, code2, code3, code4]:
        chat_interpret = interpret_history.copy()
        chat_verify = guard_history.copy()
        # refine loop
        interpretation = single_request(chat_interpret, code, model_name)
        verification = single_request(chat_verify, interpretation, model_name)

        if "VIOLATION:" in verification:
            print(verification)

        elif "ALL CONSTRAINTS SATISFIED" in verification:
            print('no violation')
            
if __name__ == "__main__":
    main()
'''