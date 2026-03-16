"""Multi-round comprehensive test combining image and function calling
using the Google GenAI SDK."""

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
    print_summary,
    print_tool_call,
    print_tool_result,
    print_user,
)
from common.prompts import COMPREHENSIVE_ROUNDS, SYSTEM_PROMPT
from common.tools import execute_tool, get_google_tools


def _process_tool_calls(client, model, contents, config):
    """Handle tool-call loops until the model produces a text response.

    Returns the final response object.
    """
    while True:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_call_parts = [p for p in parts if p.function_call]
        if not function_call_parts:
            return response

        # Append the model's tool-call response to the conversation
        contents.append(candidate.content)

        for part in function_call_parts:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)
            print_tool_call(fn_name, fn_args)

            result = execute_tool(fn_name, fn_args)
            print_tool_result(fn_name, result)

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name, response={"result": result}
                        )
                    ],
                )
            )


def main():
    client, cfg = create_google_client()
    model = cfg["model"]
    stream = cfg["stream"]

    print_header(
        "Multi-Round Comprehensive (Google GenAI)", "google_genai", model, stream
    )

    # Download image as base64
    image_url = get_image_url()
    b64_data, mime = download_image_as_base64(image_url)
    image_part = types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=get_google_tools(),
    )
    contents: list[types.Content] = []

    for i, round_info in enumerate(COMPREHENSIVE_ROUNDS, 1):
        print_round_header(i, len(COMPREHENSIVE_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        # First round includes the image
        if i == 1:
            parts = [image_part, types.Part.from_text(text=user_text)]
        else:
            parts = [types.Part.from_text(text=user_text)]

        contents.append(types.Content(role="user", parts=parts))

        # Process any tool-call loops and get the final response
        response = _process_tool_calls(client, model, contents, config)

        # Print the final text answer
        if stream:
            print_assistant(response.text)
        else:
            print_assistant(response.text)

        # Append the model's final text response to the conversation
        contents.append(response.candidates[0].content)

    print_summary()


if __name__ == "__main__":
    main()
