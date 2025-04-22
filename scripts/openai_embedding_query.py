import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

response = client.embeddings.create(
    model="bge-m3",
    input=[
        "The food was delicious and the waiter...",
        "The food was delicious and the waiter...",
    ],
)

response_dict = json.loads(response.model_dump_json())

print(response_dict.keys())

print(type(response_dict["data"]))
print(len(response_dict["data"]))
print(type(response_dict["data"][0]))
print(response_dict["data"][0].keys())
print(response_dict["data"][0]["object"])
print(response_dict["data"][0]["embedding"])
