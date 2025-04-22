import os
import time

from cicada.core.utils import cprint
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

VERBOSE = False

# Read the VERSION environment variable, default to "v3" if not set
model_version = os.getenv("VERSION", "v3")
question_level = os.getenv("LEVEL", "easy")

log_file = "deepseek_{model_version}_{timestamp}.log"
log_file_short = "deepseek_{model_version}_short_{timestamp}.log"

# List of models to test
providers = [
    "siliconflow",
    "volcengine",
    "aliyun",
    "coreshub",
    "ctyun",
    "sensecore",
    "colossal",
    "tencent",
    "baidu",
]

models = [f"deepseek-{model_version}-{each}" for each in providers]
match model_version:
    case "r1":
        models.append("deepseek-reasoner")
    case "v3":
        models.append("deepseek-chat")
    case _:
        raise ValueError("Invalid model version")

task_prompt = {
    "hard": "请找到一个三位整数 ( y )，使得 ( y + \text{reverse}(y) = 1000 )。那么 ( y ) 的值是多少？",
    "easy": "写一首雅尔塔会议的七言律诗",
}

timestamp = time.strftime("%Y%m%d%H%M%S")
log_file = log_file.format(
    model_version=model_version,
    timestamp=timestamp,
)  # Format the log file name with model version and timestamp
log_file_short = log_file_short.format(
    model_version=model_version,
    timestamp=timestamp,
)  # Format the short log file name with model version and timestamp

# Initialize the client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

log_writer = open(log_file, "w")  # Open the log file for writing
log_writer_short = open(log_file_short, "w")  # Open the short log file for writing


def verbose_print(message, color="green", **kwargs):
    if VERBOSE:
        cprint(message, color, **kwargs)


for model in models:
    print(f"Testing model: {model}")

    try:
        # Measure the start time for latency
        start_time = time.time()

        # Initiate the request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": task_prompt[question_level],
                },
            ],
            stream=True,
        )

        # Measure the end time for latency
        latency = time.time() - start_time
        verbose_print(f"Connectivity latency: {latency:.2f} seconds")

        total_tokens = 0
        start_token_time = time.time()
        last_print_time = start_token_time  # Track when we last printed token speed

        # Variables to collect content and reasoning content
        reasoning_str = ""
        common_str = ""

        # Buffer for saving all console output
        buffer = []

        verbose_print("Streaming response:")
        buffer.append("Streaming response:\n")
        complete_response = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                verbose_print(delta.content, "white", end="", flush=True)
                complete_response += delta.content
                common_str += delta.content
            # auto handle deepseek-like reasoning_content
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                verbose_print(reasoning_content, "cyan", end="", flush=True)
                complete_response += reasoning_content
                reasoning_str += reasoning_content

        # Final token speed calculation
        end_token_time = time.time()
        token_time = end_token_time - start_token_time
        total_tokens = len(complete_response)
        verbose_print(f"\nTotal Tokens: {total_tokens}")
        final_token_speed = total_tokens / token_time if token_time > 0 else 0
        print(f"\nFinal Token Speed: {final_token_speed:.2f} tokens per second")

        buffer_short = (
            f"Testing model: {model}\n"
            f"Final Token Speed: {final_token_speed:.2f} tokens per second\n"
            f"{'+'*60}\n"
        )
        log_writer_short.writelines(buffer_short)
        log_writer_short.flush()  # Ensure all data is written to the file

        # save the buffer to the log file
        buffer = (
            f"Model: {model}\n"
            f"Latency: {latency:.2f} seconds\n"
            f"Total Tokens: {total_tokens}\n"
            f"Token Speed: {final_token_speed:.2f} tokens per second\n"
            f"reasoning_response:\n{reasoning_str}\n\n"
            f"common_response:\n{common_str}\n\n"
            f"complete_response:\n{complete_response}\n\n"
            f"{'='*60}\n"
        )
        log_writer.writelines(buffer)
        log_writer.flush()  # Ensure all data is written to the file

    except Exception as e:
        print(f"Failed to get response for {model}: {str(e)}")
    print("+" * 60)
    print()


log_writer_short.close()
log_writer.close()
