"""Simple single-turn query using the Google GenAI SDK."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from common.config import create_google_client
from common.output import (
    print_assistant,
    print_header,
    print_stream_token,
    print_summary,
    print_user,
)
from common.prompts import SIMPLE_QUERY_PROMPTS, SYSTEM_PROMPT


def main():
    client, cfg = create_google_client()
    model = cfg["model"]
    stream = cfg["stream"]
    level = os.environ.get("LEVEL", "easy")

    prompt = SIMPLE_QUERY_PROMPTS.get(level, SIMPLE_QUERY_PROMPTS["easy"])
    print_header("Simple Query (Google GenAI)", "google_genai", model, stream)
    print_user(prompt)

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

    if stream:
        print("[Assistant] ", end="")
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=config
        ):
            if chunk.text:
                print_stream_token(chunk.text)
        print()
    else:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        print_assistant(response.text)

    print_summary()


if __name__ == "__main__":
    main()
