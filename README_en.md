# LLM API Simple Tests

Standalone test scripts for LLM provider APIs using official SDKs. Each provider gets 5 test scripts covering common scenarios.

## Supported Providers

| Provider | SDK | Directory |
|----------|-----|-----------|
| OpenAI Chat Completions | `openai` | `scripts/openai_chat/` |
| Anthropic Messages | `anthropic` | `scripts/anthropic/` |
| Google GenAI | `google-genai` | `scripts/google_genai/` |
| OpenAI Responses | `openai` | `scripts/openai_responses/` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API key and base URL

# Run a test
BASE_URL=https://api.openai.com/v1 API_KEY=sk-... MODEL=gpt-4o-mini \
  python scripts/openai_chat/simple_query.py

# Via gateway
BASE_URL=http://localhost:8765/v1 API_KEY=dummy MODEL=gemini-2.0-flash \
  python scripts/openai_chat/multi_round_chat.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | *required* | API base URL. Google uses `http_options.base_url` |
| `API_KEY` | *required* | API key |
| `MODEL` | `gpt-4o-mini` | Model name (set per-provider) |
| `STREAM` | `true` | Enable streaming (`true`/`false`) |
| `LEVEL` | `easy` | Prompt difficulty for `simple_query.py` (`easy`/`medium`/`hard`) |
| `TEST_IMAGE_URL` | built-in | Optional override for image test URL |

## Test Scripts

Each provider directory contains 5 scripts:

| Script | Description |
|--------|-------------|
| `simple_query.py` | Single query with configurable difficulty |
| `multi_round_chat.py` | 3-round text conversation (Fibonacci) |
| `multi_round_image.py` | 3-round image discussion |
| `multi_round_function_calling.py` | 3-round weather/temperature tool use |
| `multi_round_comprehensive.py` | Image + function calling combined |

## Directory Structure

```
llm_api_simple_tests/
├── README.md -> README_en.md
├── README_en.md
├── README_zh.md
├── requirements.txt
├── .env.example
├── assets/
│   └── test_image.jpg
└── scripts/
    ├── common/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── prompts.py
    │   ├── tools.py
    │   ├── images.py
    │   └── output.py
    ├── openai_chat/
    ├── anthropic/
    ├── google_genai/
    └── openai_responses/
```
