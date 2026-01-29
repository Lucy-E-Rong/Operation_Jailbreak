import os
import requests

# 设置 API 密钥
api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# API 端点
url = "https://api.openai.com/v1/chat/completions"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 请求体
data = {
    "model": "o3-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    #"temperature": 0.7,
    #"max_tokens": 50
    "max_completion_tokens": 50
}

# 发送请求
response = requests.post(url, headers=headers, json=data)

# 检查响应
if response.status_code == 200:
    print("API call succeeded. Response:", response.json())
else:
    print("API call failed. Error:", response.json())