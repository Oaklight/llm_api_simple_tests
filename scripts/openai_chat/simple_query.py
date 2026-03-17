"""Simple single-turn query using OpenAI Chat Completions API."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_client
from common.output import (
    print_assistant,
    print_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import SIMPLE_QUERY_PROMPTS, SYSTEM_PROMPT


def main():
    client, cfg = create_openai_client()
    stream = cfg["stream"]
    model = cfg["model"]
    level = os.environ.get("LEVEL", "easy")
    prompt = SIMPLE_QUERY_PROMPTS[level]

    print_header("Simple Query (OpenAI Chat)", "openai_chat", model, stream)
    print_user(prompt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

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

    print_summary(success=bool(full_text))


if __name__ == "__main__":
    main()
