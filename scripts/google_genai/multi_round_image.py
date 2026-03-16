"""Multi-round image chat using the Google GenAI SDK."""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from common.config import create_google_client
from common.images import download_image_as_base64, get_image_url
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
    client, cfg = create_google_client()
    model = cfg["model"]
    stream = cfg["stream"]

    print_header("Multi-Round Image Chat (Google GenAI)", "google_genai", model, stream)

    # Download image as base64 (URL-based image parts are not supported)
    image_url = get_image_url()
    b64_data, mime = download_image_as_base64(image_url)
    image_part = types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime)

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    contents: list[types.Content] = []

    for i, round_info in enumerate(IMAGE_ROUNDS, 1):
        print_round_header(i, len(IMAGE_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        # First round includes the image; subsequent rounds are text-only
        if i == 1:
            parts = [image_part, types.Part.from_text(text=user_text)]
        else:
            parts = [types.Part.from_text(text=user_text)]

        contents.append(types.Content(role="user", parts=parts))

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
