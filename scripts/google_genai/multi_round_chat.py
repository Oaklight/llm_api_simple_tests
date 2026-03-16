"""Multi-round chat using the Google GenAI SDK."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from common.config import create_google_client
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
    client, cfg = create_google_client()
    model = cfg["model"]
    stream = cfg["stream"]

    print_header("Multi-Round Chat (Google GenAI)", "google_genai", model, stream)

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    contents: list[types.Content] = []

    for i, round_info in enumerate(CHAT_ROUNDS, 1):
        print_round_header(i, len(CHAT_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

        if stream:
            print("[Assistant] ", end="")
            full_text = ""
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                if chunk.text:
                    print_stream_token(chunk.text)
                    full_text += chunk.text
            print()
            # Append the model response to maintain conversation history
            contents.append(
                types.Content(
                    role="model", parts=[types.Part.from_text(text=full_text)]
                )
            )
        else:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            print_assistant(response.text)
            contents.append(response.candidates[0].content)

    print_summary()


if __name__ == "__main__":
    main()
