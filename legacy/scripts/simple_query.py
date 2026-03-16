import os

from cicada.core.model import MultiModalModel
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

model_name = os.getenv("MODEL", "deepseek-v3")
question_level = os.getenv("LEVEL", "easy")
stream = os.getenv("STREAM", "True").lower() == "true"  # Configurable stream option

task_prompt = {
    "hard": "请找到一个三位整数 ( y )，使得 ( y + \text{reverse}(y) = 1000 )。那么 ( y ) 的值是多少？",
    "easy": "写一首雅尔塔会议的七言律诗",
    "medium": "等红灯其实是等绿灯",
}

model = MultiModalModel(
    api_key=API_KEY,
    api_base_url=BASE_URL,
    model_name=model_name,
    stream=stream,
)

response = model.query(
    task_prompt[question_level],
    stream=stream,
)
if not stream:
    print(response)
