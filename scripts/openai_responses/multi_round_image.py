"""Multi-round image conversation using the OpenAI Responses API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_responses_client
from common.images import get_image_url
from common.output import (
    print_assistant,
    print_header,
    print_round_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import IMAGE_ROUNDS, SYSTEM_PROMPT


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
    image_url = get_image_url()

    print_header(
        "Multi-Round Image Chat (OpenAI Responses API)",
        "openai_responses",
        model,
        stream,
    )
    print(f"  Image URL: {image_url}")

    input_items: list = []

    for i, round_info in enumerate(IMAGE_ROUNDS, 1):
        print_round_header(i, len(IMAGE_ROUNDS))
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
            # Subsequent rounds: text only
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
            print()

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

            for item in response.output:
                input_items.append(item)

    print_summary(success=True)


if __name__ == "__main__":
    main()
