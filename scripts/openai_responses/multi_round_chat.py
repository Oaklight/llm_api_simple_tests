"""Multi-round chat using the OpenAI Responses API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_responses_client
from common.output import (
    print_assistant,
    print_header,
    print_round_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import CHAT_ROUNDS, SYSTEM_PROMPT


def extract_text(response) -> str:
    """Extract assistant text from a non-streamed response."""
    text = ""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text += content.text
    return text


def main():
    client, cfg = create_openai_responses_client()
    model = cfg["model"]
    stream = cfg["stream"]

    print_header(
        "Multi-Round Chat (OpenAI Responses API)", "openai_responses", model, stream
    )

    input_items: list = []

    for i, round_info in enumerate(CHAT_ROUNDS, 1):
        print_round_header(i, len(CHAT_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        input_items.append({"role": "user", "content": user_text})

        if stream:
            response_stream = client.responses.create(
                model=model,
                input=input_items,
                instructions=SYSTEM_PROMPT,
                stream=True,
            )
            print("[Assistant] ", end="")
            text = ""
            completed_response = None
            for event in response_stream:
                if event.type == "response.output_text.delta":
                    print_stream_token(event.delta)
                    text += event.delta
                elif event.type == "response.completed":
                    completed_response = event.response
            print()  # newline after streaming

            # Add the completed response output items to input for next round
            if completed_response is not None:
                input_items.extend(completed_response.output)
        else:
            response = client.responses.create(
                model=model,
                input=input_items,
                instructions=SYSTEM_PROMPT,
            )
            text = extract_text(response)
            print_assistant(text)

            # Add response output items to input for next round
            for item in response.output:
                input_items.append(item)

    print_summary(success=True)


if __name__ == "__main__":
    main()
