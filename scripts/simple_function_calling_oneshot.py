import os

from dotenv import load_dotenv
from openai import OpenAI
from toolregistry import ToolRegistry

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

tool_registry = ToolRegistry()


# 注册工具
@tool_registry.register
def get_weather(location: str):
    return f"Weather in {location}: Sunny, 25°C"


@tool_registry.register
def c_to_f(celsius: float) -> float:
    fahrenheit = (celsius * 1.8) + 32
    return f"{celsius} celsius degree == {fahrenheit} fahrenheit degree"


response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tool_registry.get_tools_json(),
    tool_choice="auto",
)

tool_calls = response.choices[0].message.tool_calls
# print(f"Function called: {tool_call.name}")
# print(f"Arguments: {tool_call.arguments}")
# print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")

print(f"tool_calls: {tool_calls}")

content = response.choices[0].message.content
print(f"content: {content}")
