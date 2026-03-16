"""Multi-round comprehensive test: image + function calling using the Anthropic SDK."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_anthropic_client
from common.images import get_image_url
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
from common.prompts import COMPREHENSIVE_ROUNDS, SYSTEM_PROMPT
from common.tools import execute_tool, get_anthropic_tools


def _handle_tool_calls(client, model, messages, tools, response):
    """Process tool calls in a loop until the model produces a final text response."""
    while True:
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        # Append the full assistant response to messages
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool and build tool_result blocks
        tool_results = []
        for block in tool_use_blocks:
            print_tool_call(block.name, block.input)
            result = execute_tool(block.name, block.input)
            print_tool_result(block.name, result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

        # Call the API again to let the model process tool results
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

    return response


def main():
    client, cfg = create_anthropic_client()
    model = cfg["model"]
    stream = cfg["stream"]
    tools = get_anthropic_tools()
    image_url = get_image_url()

    print_header("Multi-Round Comprehensive (Anthropic)", "Anthropic", model, stream)

    messages = []
    total = len(COMPREHENSIVE_ROUNDS)

    for i, round_data in enumerate(COMPREHENSIVE_ROUNDS, 1):
        print_round_header(i, total)
        round_text = round_data["content"]
        print_user(round_text)

        # Build user content: first round includes image, rest are text-only
        if i == 1:
            user_content = [
                {
                    "type": "image",
                    "source": {"type": "url", "url": image_url},
                },
                {"type": "text", "text": round_text},
            ]
        else:
            user_content = round_text

        messages.append({"role": "user", "content": user_content})

        # Initial call is non-streaming (may produce tool calls)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        # Handle any tool call loops
        response = _handle_tool_calls(client, model, messages, tools, response)

        # Extract the final text reply
        text_blocks = [b for b in response.content if b.type == "text"]
        reply = text_blocks[0].text if text_blocks else ""

        if stream and not text_blocks:
            print("[Assistant] ", end="")
            collected = []
            with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            ) as s:
                for text in s.text_stream:
                    print_stream_token(text)
                    collected.append(text)
            print()
            reply = "".join(collected)
        else:
            print_assistant(reply)

        messages.append({"role": "assistant", "content": reply})

    print_summary(success=True)


if __name__ == "__main__":
    main()
