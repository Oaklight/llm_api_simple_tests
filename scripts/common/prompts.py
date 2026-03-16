"""Centralized prompt definitions for all test scripts."""

SYSTEM_PROMPT = "You are a helpful assistant."

# --- simple_query.py ---

SIMPLE_QUERY_PROMPTS = {
    "easy": "What is 2 + 2?",
    "medium": "Explain the difference between a list and a tuple in Python.",
    "hard": (
        "Compare and contrast the architectural approaches of microservices "
        "and monolithic applications. Discuss trade-offs in scalability, "
        "maintainability, and deployment complexity."
    ),
}

# --- multi_round_chat.py ---

CHAT_ROUNDS = [
    {"role": "user", "content": "Explain the Fibonacci sequence in simple terms."},
    {
        "role": "user",
        "content": "Can you write a Python function to compute the nth Fibonacci number?",
    },
    {
        "role": "user",
        "content": "What is the time complexity of that function? Can you optimize it?",
    },
]

# --- multi_round_image.py ---

IMAGE_ROUNDS = [
    {"role": "user", "content": "Describe what you see in this image in detail."},
    {
        "role": "user",
        "content": "Based on the image, can you guess where this might be located?",
    },
    {
        "role": "user",
        "content": "Tell me some interesting facts about this place or landmark.",
    },
]

# --- multi_round_function_calling.py ---

FUNCTION_CALLING_ROUNDS = [
    {"role": "user", "content": "What's the current weather in San Francisco?"},
    {"role": "user", "content": "Convert that temperature to Celsius."},
    {
        "role": "user",
        "content": "Now check the weather in Tokyo and compare it to San Francisco.",
    },
]

# --- multi_round_comprehensive.py ---

COMPREHENSIVE_ROUNDS = [
    {"role": "user", "content": "Describe the landmark in this image."},
    {"role": "user", "content": "Check the current weather at that location."},
    {
        "role": "user",
        "content": "Based on the landmark and the weather, give me a travel recommendation.",
    },
]
