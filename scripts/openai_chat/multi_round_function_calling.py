"""Multi-round function calling using OpenAI Chat Completions API."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_client
from common.output import (
    print_assistant,
    print_header,
    print_round_header,
    print_stream_token,
    print_summary,
    print_tool_call,
    print_tool_result,
    print_user,
)
from common.prompts import FUNCTION_CALLING_ROUNDS, SYSTEM_PROMPT
from common.tools import execute_tool, get_openai_chat_tools


def handle_tool_calls(client, model, messages, tools, stream):
    """Make an API call, handle any tool calls, and return the final text."""
    # Initial call is always non-streaming to reliably capture tool calls.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=False,
    )
    assistant_message = response.choices[0].message

    # If no tool calls, return the text directly.
    if not assistant_message.tool_calls:
        text = assistant_message.content or ""
        print_assistant(text)
        messages.append({"role": "assistant", "content": text})
        return text

    # Process tool calls.
    messages.append(assistant_message)

    for tc in assistant_message.tool_calls:
        args = json.loads(tc.function.arguments)
        print_tool_call(tc.function.name, args)
        result = execute_tool(tc.function.name, args)
        print_tool_result(tc.function.name, result)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )

    # Follow-up call to get a text response after tool results.
    if stream:
        follow_up = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
        )
        print("[Assistant] ", end="")
        full_text = ""
        for chunk in follow_up:
            token = chunk.choices[0].delta.content
            if token:
                print_stream_token(token)
                full_text += token
        print()
    else:
        follow_up = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        full_text = follow_up.choices[0].message.content or ""
        print_assistant(full_text)

    messages.append({"role": "assistant", "content": full_text})
    return full_text


def main():
    client, cfg = create_openai_client()
    stream = cfg["stream"]
    model = cfg["model"]
    tools = get_openai_chat_tools()

    print_header(
        "Multi-Round Function Calling (OpenAI Chat)", "openai_chat", model, stream
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total = len(FUNCTION_CALLING_ROUNDS)
    full_text = ""

    for i, round_info in enumerate(FUNCTION_CALLING_ROUNDS, 1):
        print_round_header(i, total)
        print_user(round_info["content"])

        messages.append({"role": "user", "content": round_info["content"]})
        full_text = handle_tool_calls(client, model, messages, tools, stream)

    print_summary(success=bool(full_text))


if __name__ == "__main__":
    main()
