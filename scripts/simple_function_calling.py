import os

from cicada.core.model import MultiModalModel
from cicada.core.utils import cprint
from dotenv import load_dotenv
from toolregistry import ToolRegistry

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

model_name = os.getenv("MODEL", "deepseek-v3")
stream = os.getenv("STREAM", "True").lower() == "true"  # Configurable stream option

llm = MultiModalModel(
    api_key=API_KEY,
    api_base_url=BASE_URL,
    model_name=model_name,
    stream=stream,
)


tool_registry = ToolRegistry()


# 注册工具
@tool_registry.register
def get_weather(location: str):
    return f"Weather in {location}: Sunny, 25°C"


@tool_registry.register
def c_to_f(celsius: float) -> float:
    fahrenheit = (celsius * 1.8) + 32
    return f"{celsius} celsius degree == {fahrenheit} fahrenheit degree"


# 使用工具调用
response = llm.query(
    "How's weather in Chicago? Answer in Fahrenheit.",
    tools=tool_registry,
    stream=stream,
)
print(response["content"])

cprint(response)
