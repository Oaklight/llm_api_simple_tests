"""Tool specifications in all 4 provider formats and mock implementations."""

import json


# ============================================================================
# Mock tool implementations
# ============================================================================


def get_weather(location: str) -> dict:
    """Return mock weather data for a location."""
    data = {
        "san francisco": {
            "temperature": 72,
            "unit": "fahrenheit",
            "condition": "sunny",
            "humidity": "45%",
        },
        "tokyo": {
            "temperature": 15,
            "unit": "celsius",
            "condition": "cloudy",
            "humidity": "65%",
        },
        "paris": {
            "temperature": 18,
            "unit": "celsius",
            "condition": "partly cloudy",
            "humidity": "55%",
        },
    }
    key = location.lower().strip()
    for city, info in data.items():
        if city in key:
            return {"location": location, **info}
    return {
        "location": location,
        "temperature": 20,
        "unit": "celsius",
        "condition": "clear",
        "humidity": "50%",
    }


def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert temperature between Fahrenheit and Celsius."""
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    if from_unit == to_unit:
        return {"value": value, "unit": to_unit}
    if from_unit in ("f", "fahrenheit"):
        converted = (value - 32) * 5 / 9
    else:
        converted = value * 9 / 5 + 32
    return {
        "original": value,
        "original_unit": from_unit,
        "converted": round(converted, 1),
        "converted_unit": to_unit,
    }


def execute_tool(name: str, args: dict) -> str:
    """Dispatch a tool call and return JSON result."""
    if name == "get_weather":
        return json.dumps(get_weather(args.get("location", "Unknown")))
    elif name == "convert_temperature":
        return json.dumps(
            convert_temperature(
                float(args.get("value", 0)),
                args.get("from_unit", "fahrenheit"),
                args.get("to_unit", "celsius"),
            )
        )
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ============================================================================
# Tool specs: OpenAI Chat format
# ============================================================================

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name, e.g. San Francisco",
        },
    },
    "required": ["location"],
}

_CONVERT_TEMP_PARAMS = {
    "type": "object",
    "properties": {
        "value": {"type": "number", "description": "Temperature value to convert"},
        "from_unit": {
            "type": "string",
            "enum": ["fahrenheit", "celsius"],
            "description": "Source unit",
        },
        "to_unit": {
            "type": "string",
            "enum": ["fahrenheit", "celsius"],
            "description": "Target unit",
        },
    },
    "required": ["value", "from_unit", "to_unit"],
}


def get_openai_chat_tools() -> list:
    """OpenAI Chat Completions tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": _WEATHER_PARAMS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_temperature",
                "description": "Convert temperature between Fahrenheit and Celsius",
                "parameters": _CONVERT_TEMP_PARAMS,
            },
        },
    ]


def get_anthropic_tools() -> list:
    """Anthropic tool format (uses input_schema instead of parameters)."""
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": _WEATHER_PARAMS,
        },
        {
            "name": "convert_temperature",
            "description": "Convert temperature between Fahrenheit and Celsius",
            "input_schema": _CONVERT_TEMP_PARAMS,
        },
    ]


def get_google_tools() -> list:
    """Google GenAI tool format."""
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get the current weather in a given location",
                    parameters=_WEATHER_PARAMS,
                ),
                types.FunctionDeclaration(
                    name="convert_temperature",
                    description="Convert temperature between Fahrenheit and Celsius",
                    parameters=_CONVERT_TEMP_PARAMS,
                ),
            ]
        )
    ]


def get_openai_responses_tools() -> list:
    """OpenAI Responses API tool format (flatter structure)."""
    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": _WEATHER_PARAMS,
        },
        {
            "type": "function",
            "name": "convert_temperature",
            "description": "Convert temperature between Fahrenheit and Celsius",
            "parameters": _CONVERT_TEMP_PARAMS,
        },
    ]
