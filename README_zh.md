# LLM API 简单测试

[![Last Commit](https://img.shields.io/github/last-commit/Oaklight/llm_api_simple_tests?color=green)](https://github.com/Oaklight/llm_api_simple_tests)
[![License](https://img.shields.io/github/license/Oaklight/llm_api_simple_tests?color=green)](https://github.com/Oaklight/llm_api_simple_tests/blob/master/LICENSE)

使用官方 SDK 的 LLM 提供商 API 独立测试脚本。每个提供商有 5 个测试脚本，覆盖常见场景。

## 支持的提供商

| 提供商 | SDK | 目录 |
|--------|-----|------|
| OpenAI Chat Completions | `openai` | `scripts/openai_chat/` |
| Anthropic Messages | `anthropic` | `scripts/anthropic/` |
| Google GenAI | `google-genai` | `scripts/google_genai/` |
| OpenAI Responses | `openai` | `scripts/openai_responses/` |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API key 和 base URL

# 运行测试
BASE_URL=https://api.openai.com/v1 API_KEY=sk-... MODEL=gpt-4o-mini \
  python scripts/openai_chat/simple_query.py

# 通过网关运行
BASE_URL=http://localhost:8765/v1 API_KEY=dummy MODEL=gemini-2.0-flash \
  python scripts/openai_chat/multi_round_chat.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASE_URL` | *必填* | API 基础 URL。Google 使用 `http_options.base_url` |
| `API_KEY` | *必填* | API 密钥 |
| `MODEL` | `gpt-4o-mini` | 模型名称（根据提供商设置） |
| `STREAM` | `true` | 启用流式输出（`true`/`false`） |
| `LEVEL` | `easy` | `simple_query.py` 的提示难度（`easy`/`medium`/`hard`） |
| `TEST_IMAGE_URL` | 内置 | 可选的图片测试 URL 覆盖 |

## 测试脚本

每个提供商目录包含 5 个脚本：

| 脚本 | 描述 |
|------|------|
| `simple_query.py` | 单次查询，可配置难度 |
| `multi_round_chat.py` | 3 轮文本对话（斐波那契） |
| `multi_round_image.py` | 3 轮图片讨论 |
| `multi_round_function_calling.py` | 3 轮天气/温度工具调用 |
| `multi_round_comprehensive.py` | 图片 + 函数调用综合测试 |

## 目录结构

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
