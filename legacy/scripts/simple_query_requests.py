import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
# Configuration
model_name = os.getenv("MODEL", "deepseek-v3")
question_level = os.getenv("LEVEL", "easy")
stream = os.getenv("STREAM", "True").lower() == "true"  # Configurable stream option

task_prompt = {
    "hard": "请找到一个三位整数 ( y )，使得 ( y + \text{reverse}(y) = 1000 )。那么 ( y ) 的值是多少？",
    "easy": "写一首雅尔塔会议的七言律诗",
    "medium": "等红灯其实是等绿灯",
}


model_endpoint = f"{BASE_URL}/chat/completions"

# Prepare the request payload
payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": task_prompt[question_level]}],
    "stream": stream,
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Make the request
response = requests.post(model_endpoint, json=payload, headers=headers, stream=stream)

# Handle the response based on the stream option
if stream:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode("utf-8"), end="")
else:
    print(response.json())
