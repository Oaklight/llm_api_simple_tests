import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."


tool_functions = {"get_weather": get_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    }
]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)


tool_call = response.choices[0].message.tool_calls
# print(f"Function called: {tool_call.name}")
# print(f"Arguments: {tool_call.arguments}")
# print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")

print(tool_call)

content = response.choices[0].message.content
print(content)
