# implement LLM simulator for defense
#整合到airsim_gpt_demo.py中
import numpy as np
import datetime
from openai import OpenAI

with open("system_prompts/constraint_interpretation.txt", "r") as f:
    interpret_sysprompt = f.read()
with open("system_prompts/constraint_verification.txt", "r") as f:
    guard_sysprompt = f.read()


print("Initializing GPT...")
client = OpenAI()

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


def request(history, prompt, model_name):
    history.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    completion = client.chat.completions.create(
        model = model_name, # "gpt-4", "gpt-4o-mini", "gpt-4-turbo"
        messages = history
    )
    response = completion.choices[0].message.content
    history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
    return response

def single_request(history, prompt, model_name):
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

model_name = "gpt-4o" # "o3-mini", "o4-mini", "o3", "gpt-5", "gpt-4.1"
print(f"Guard using {model_name}")

chat_interpret = interpret_history.copy()
chat_verify = guard_history.copy()
# refine loop
interpretation = single_request(chat_interpret, code, model_name)  #code 为原本airsim生成内容
verification = single_request(chat_verify, interpretation, model_name)

if "VIOLATION:" in verification:
    print(verification)

elif "ALL CONSTRAINTS SATISFIED" in verification:
    print('no violation')


