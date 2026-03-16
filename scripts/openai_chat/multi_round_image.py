"""Multi-round image chat using OpenAI Chat Completions API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_client
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


def main():
    client, cfg = create_openai_client()
    stream = cfg["stream"]
    model = cfg["model"]
    image_url = get_image_url()

    print_header("Multi-Round Image Chat (OpenAI Chat)", "openai_chat", model, stream)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total = len(IMAGE_ROUNDS)

    for i, round_info in enumerate(IMAGE_ROUNDS, 1):
        print_round_header(i, total)
        print_user(round_info["content"])

        if i == 1:
            user_content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": round_info["content"]},
            ]
        else:
            user_content = round_info["content"]

        messages.append({"role": "user", "content": user_content})

        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            print("[Assistant] ", end="")
            full_text = ""
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    print_stream_token(token)
                    full_text += token
            print()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            full_text = response.choices[0].message.content
            print_assistant(full_text)

        messages.append({"role": "assistant", "content": full_text})

    print_summary(success=bool(full_text))


if __name__ == "__main__":
    main()
