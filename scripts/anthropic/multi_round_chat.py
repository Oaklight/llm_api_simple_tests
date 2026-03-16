"""Multi-round chat conversation using the Anthropic SDK."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_anthropic_client
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
    client, cfg = create_anthropic_client()
    model = cfg["model"]
    stream = cfg["stream"]

    print_header("Multi-Round Chat (Anthropic)", "Anthropic", model, stream)

    messages = []
    total = len(CHAT_ROUNDS)

    for i, round_data in enumerate(CHAT_ROUNDS, 1):
        print_round_header(i, total)
        user_text = round_data["content"]
        print_user(user_text)

        messages.append({"role": "user", "content": user_text})

        if stream:
            print("[Assistant] ", end="")
            collected = []
            with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as s:
                for text in s.text_stream:
                    print_stream_token(text)
                    collected.append(text)
            print()
            reply = "".join(collected)
        else:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            reply = response.content[0].text
            print_assistant(reply)

        messages.append({"role": "assistant", "content": reply})

    print_summary(success=True)


if __name__ == "__main__":
    main()
