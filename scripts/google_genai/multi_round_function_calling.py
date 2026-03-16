"""Multi-round function calling using the Google GenAI SDK."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from common.config import create_google_client
from common.output import (
    print_assistant,
    print_header,
    print_round_header,
    print_summary,
    print_tool_call,
    print_tool_result,
    print_user,
)
from common.prompts import FUNCTION_CALLING_ROUNDS, SYSTEM_PROMPT
from common.tools import execute_tool, get_google_tools


def _process_tool_calls(client, model, contents, config):
    """Handle tool-call loops until the model produces a text response.

    Returns the final text response string.
    """
    while True:
        # Non-streaming call for tool-call detection
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check whether any part is a function call
        function_call_parts = [p for p in parts if p.function_call]
        if not function_call_parts:
            # No tool calls — return the text response
            return response

        # Append the model's response (contains function_call parts)
        contents.append(candidate.content)

        # Execute each tool call and send results back
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
        "Multi-Round Function Calling (Google GenAI)", "google_genai", model, stream
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=get_google_tools(),
    )
    contents: list[types.Content] = []

    for i, round_info in enumerate(FUNCTION_CALLING_ROUNDS, 1):
        print_round_header(i, len(FUNCTION_CALLING_ROUNDS))
        user_text = round_info["content"]
        print_user(user_text)

        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

        # Process any tool-call loops and get the final response
        response = _process_tool_calls(client, model, contents, config)

        # Print the final text answer
        if stream:
            # The text response is already obtained non-streaming from the
            # tool-call loop; just print it.  If the caller wants a streamed
            # final answer we could re-issue, but the response is already
            # available.
            print_assistant(response.text)
        else:
            print_assistant(response.text)

        # Append the model's final text response to the conversation
        contents.append(response.candidates[0].content)

    print_summary()


if __name__ == "__main__":
    main()
