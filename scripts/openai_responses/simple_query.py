"""Simple single-turn query using the OpenAI Responses API."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import create_openai_responses_client
from common.output import (
    print_assistant,
    print_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import SIMPLE_QUERY_PROMPTS, SYSTEM_PROMPT


def run_non_streaming(client, model: str, prompt: str) -> str:
    """Send a single query without streaming and return the response text."""
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        instructions=SYSTEM_PROMPT,
    )
    text = ""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text += content.text
    return text


def run_streaming(client, model: str, prompt: str) -> str:
    """Send a single query with streaming and return the full response text."""
    stream = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        instructions=SYSTEM_PROMPT,
        stream=True,
    )
    text = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            print_stream_token(event.delta)
            text += event.delta
    print()  # newline after streaming
    return text


def main():
    client, cfg = create_openai_responses_client()
    model = cfg["model"]
    stream = cfg["stream"]

    level = os.environ.get("LEVEL", "easy").lower()
    prompt = SIMPLE_QUERY_PROMPTS.get(level, SIMPLE_QUERY_PROMPTS["easy"])

    print_header(
        "Simple Query (OpenAI Responses API)", "openai_responses", model, stream
    )
    print_user(prompt)

    if stream:
        print("[Assistant] ", end="")
        text = run_streaming(client, model, prompt)
    else:
        text = run_non_streaming(client, model, prompt)
        print_assistant(text)

    print_summary(success=bool(text.strip()))


if __name__ == "__main__":
    main()
