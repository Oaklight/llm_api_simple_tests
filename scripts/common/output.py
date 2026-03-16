"""Print helpers for formatted output."""

import sys


def print_header(title: str, provider: str, model: str, stream: bool) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Provider: {provider}  |  Model: {model}  |  Stream: {stream}")
    print(f"{'=' * 60}")


def print_round_header(round_num: int, total: int) -> None:
    print(f"\n--- Round {round_num}/{total} ---")


def print_user(content: str) -> None:
    print(f"[User] {content}")


def print_assistant(content: str) -> None:
    print(f"[Assistant] {content}")


def print_tool_call(name: str, args: dict) -> None:
    print(f"[Tool Call] {name}({args})")


def print_tool_result(name: str, result: str) -> None:
    print(f"[Tool Result] {name} -> {result}")


def print_stream_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def print_summary(success: bool = True) -> None:
    status = "PASSED" if success else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"  Result: {status}")
    print(f"{'=' * 60}\n")
