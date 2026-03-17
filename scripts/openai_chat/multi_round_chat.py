"""Multi-round text chat using OpenAI Chat Completions API."""

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
    print_user,
)
from common.prompts import CHAT_ROUNDS, SYSTEM_PROMPT


def main():
    client, cfg = create_openai_client()
    stream = cfg["stream"]
    model = cfg["model"]

    print_header("Multi-Round Chat (OpenAI Chat)", "openai_chat", model, stream)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total = len(CHAT_ROUNDS)

    for i, round_info in enumerate(CHAT_ROUNDS, 1):
        print_round_header(i, total)
        print_user(round_info["content"])

        messages.append({"role": "user", "content": round_info["content"]})

        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            print("[Assistant] ", end="")
            full_text = ""
            for chunk in response:
                if not chunk.choices:
                    continue
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
