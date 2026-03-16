"""Multi-round comprehensive test combining image and function calling
using the OpenAI Responses API."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_responses_client
from common.images import get_image_url
from common.output import (
    print_assistant,
    print_header,
    print_round_header,
    print_summary,
    print_tool_call,
    print_tool_result,
    print_user,
)
from common.prompts import COMPREHENSIVE_ROUNDS, SYSTEM_PROMPT
from common.tools import execute_tool, get_openai_responses_tools


def extract_text(response) -> str:
    """Extract assistant text from a non-streamed response."""
    text = ""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text += content.text
    return text


def process_tool_calls(response, input_items: list) -> bool:
    """Process any tool calls in the response. Returns True if tool calls were found."""
    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return False

    # Add all response output items (including function_call items) to input
    for item in response.output:
        input_items.append(item)

    # Execute each tool call and append results
    for tc in tool_calls:
        args = json.loads(tc.arguments)
        print_tool_call(tc.name, args)
        result = execute_tool(tc.name, args)
        print_tool_result(tc.name, result)

        input_items.append(
            {
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": result,
            }
        )

    return True


def main():
    client, cfg = create_openai_responses_client()
    model = cfg["model"]
    stream = cfg["stream"]
    tools = get_openai_responses_tools()
    image_url = get_image_url()

    print_header(
        "Multi-Round Comprehensive (OpenAI Responses API)",
        "openai_responses",
        model,
        stream,
    )
    print(f"  Image URL: {image_url}")

    input_items: list = []

    for i, round_info in enumerate(COMPREHENSIVE_ROUNDS, 1):
        print_round_header(i, len(COMPREHENSIVE_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        if i == 1:
            # First round: include image alongside text
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": user_text},
                    ],
                }
            )
        else:
            input_items.append({"role": "user", "content": user_text})

        # Always use non-streaming for the initial request to capture tool calls
        response = client.responses.create(
            model=model,
            input=input_items,
            instructions=SYSTEM_PROMPT,
            tools=tools,
        )

        # Process tool calls in a loop until we get a text response
        while process_tool_calls(response, input_items):
            response = client.responses.create(
                model=model,
                input=input_items,
                instructions=SYSTEM_PROMPT,
                tools=tools,
            )

        # Print the text response
        text = extract_text(response)
        print_assistant(text)

        # Add final response output to input for next round
        for item in response.output:
            input_items.append(item)

    print_summary(success=True)


if __name__ == "__main__":
    main()
