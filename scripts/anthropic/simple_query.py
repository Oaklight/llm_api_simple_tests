"""Simple single-turn query using the Anthropic SDK."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_anthropic_client
from common.output import (
    print_assistant,
    print_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import SIMPLE_QUERY_PROMPTS, SYSTEM_PROMPT


def main():
    client, cfg = create_anthropic_client()
    model = cfg["model"]
    stream = cfg["stream"]
    level = os.environ.get("LEVEL", "easy").lower()
    prompt = SIMPLE_QUERY_PROMPTS.get(level, SIMPLE_QUERY_PROMPTS["easy"])

    print_header("Simple Query (Anthropic)", "Anthropic", model, stream)
    print_user(prompt)

    messages = [{"role": "user", "content": prompt}]

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
        print()  # newline after stream
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

    print_summary(success=bool(reply))


if __name__ == "__main__":
    main()
