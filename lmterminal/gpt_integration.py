import sys
import time

import openai
import tiktoken

_client = None


def _get_client(api_key: str) -> openai.OpenAI:
    """Return a reusable OpenAI client."""
    global _client
    if _client is None or _client.api_key != api_key:
        _client = openai.OpenAI(api_key=api_key)
    return _client


BLUE = "\x1b[34m"
RED = "\x1b[91m"
RESET = "\x1b[0m"

DEFAULT_MODEL = "gpt-5-nano"
STREAMING_REASONING_EFFORT = "minimal"


def format_prompt(system_content, user_content):
    """Returns a formatted prompt for the OpenAI API."""
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def chatgpt_request(
    api_key,
    prompt,
    model=DEFAULT_MODEL,
    # max_tokens=3900,
    n=1,
    temperature=1,
    stop=None,
    stream=False,
    update_markdown_stream=None,
):
    """
    Sends a request to the OpenAI Chat API.

    Returns:
        tuple[str, float, object]:
            * generated_text
            * response_time (seconds)
            * raw_response (non-stream) or collected stream chunks (stream)
    """
    start_time = time.monotonic_ns()

    client = _get_client(api_key)

    request_kwargs = {
        "messages": prompt,
        "model": model,
        "n": n,
        "temperature": temperature,
        "stream": stream,
    }
    if stream and model.startswith("gpt-5"):
        # `gpt-5*` defaults to heavier reasoning that delays the first token.
        # Force minimal reasoning effort for streamed output so users see live updates.
        request_kwargs["reasoning_effort"] = STREAMING_REASONING_EFFORT

    # Make the API request
    response = client.chat.completions.create(**request_kwargs)

    if stream:
        # Create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []

        # Iterate through the stream of events
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            delta = chunk.choices[0].delta  # extract the delta
            if delta.content is not None:
                collected_messages.append(delta.content)  # save the message

            if update_markdown_stream:
                update_markdown_stream(delta.content or "")
            else:
                print(delta.content or "", end="", flush=True)

        # Save the time delay and text received
        response_time = (time.monotonic_ns() - start_time) / 1e9
        generated_text = "".join(collected_messages)
        response_payload = collected_chunks

    else:
        # Extract and save the generated response
        generated_text = response.choices[0].message.content

        # Save the time delay
        response_time = (time.monotonic_ns() - start_time) / 1e9
        response_payload = response

    return (
        generated_text,
        response_time,
        response_payload,
    )


def num_tokens_from_string(string, model=DEFAULT_MODEL):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model=DEFAULT_MODEL):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def estimated_cost(num_tokens, price_per_1M_tokens):
    """Returns the estimated cost of a number of tokens."""
    return f"{num_tokens / 10**6 * price_per_1M_tokens:.6f}"


def estimate_prompt_cost(message, model):
    """Returns the estimated cost of a prompt."""
    num_tokens = num_tokens_from_messages(message, model)

    # Prices in USD per 1M input tokens
    prices = {
        "gpt-3.5-turbo": 0.50,
        "gpt-3.5-turbo-0125": 0.50,
        "gpt-3.5-turbo-1106": 0.50,
        "gpt-3.5-turbo-instruct": 1.50,
        "gpt-4": 30,
        "gpt-4-turbo-preview": 10,
        "gpt-4-turbo": 10,
        "gpt-4-turbo-2024-04-09": 0.01,
        "gpt-4-0613": 0.03,
        "gpt-4-1106-preview": 10,
        "gpt-4-0125-preview": 10,
        "gpt-4-32k": 60,
        "gpt-4-32k-0613": 60,
        "gpt-4o": 2.50,
        "gpt-4o-2024-05-13": 5,
        "gpt-4o-2024-08-06": 2.50,
        "gpt-4o-2024-11-20": 2.50,
        "gpt-4o-mini": 0.15,
        "gpt-4o-mini-2024-07-18": 0.15,
        "chatgpt-4o-latest": 5,
        "o1": 15,
        "o1-2024-12-17": 15,
        "o1-preview": 15,
        "o1-preview-2024-09-12": 15,
        "o1-mini": 1.10,
        "o1-mini-2024-09-12": 1.10,
        "o1-pro": 150,
        "o1-pro-2025-03-19": 150,
        "gpt-4.1": 2,
        "gpt-4.1-2025-04-14": 2,
        "gpt-4.1-mini": 0.40,
        "gpt-4.1-mini-2025-04-14": 0.40,
        "gpt-4.1-nano": 0.1,
        "gpt-4.1-nano-2025-04-14": 0.10,
        "gpt-4.5-preview": 75,
        "o3": 2,
        "o3-2025-04-16": 2,
        "o3-mini": 1.10,
        "o3-mini-2025-01-31": 1.10,
        "o4-mini": 1.10,
        "o4-mini-2025-04-16": 1.10,
        "codex-mini-latest": 1.50,
        "gpt-4o-search-preview": 2.50,
        "gpt-4o-search-preview-2025-03-11": 2.50,
        "gpt-4o-mini-search-preview": 0.15,
        "gpt-4o-mini-search-preview-2025-03-11": 0.15,
        "gpt-5": 1.25,
        "gpt-5-mini": 0.25,
        "gpt-5-nano": 0.05,
        "gpt-5-chat-latest": 1.25,
        "gpt-5.1": 1.25,
        "gpt-5.2": 1.75,
    }

    return estimated_cost(num_tokens, prices[model])


def handle_rate_limit_error():
    """
    Provides guidance on how to handle a rate limit error.
    """
    sys.stderr.write("\n")
    sys.stderr.write(
        BLUE
        + "You might not have set a usage rate limit in your OpenAI account settings. "
        + RESET
        + "\n"
    )
    sys.stderr.write(
        "If that's the case, you can set it"
        " here:\nhttps://platform.openai.com/account/billing/limits" + "\n"
    )

    sys.stderr.write("\n")
    sys.stderr.write(
        BLUE + "If you have set a usage rate limit, please try the following steps:" + RESET + "\n"
    )
    sys.stderr.write("- Wait a few seconds before trying again.\n")
    sys.stderr.write("\n")
    sys.stderr.write(
        "- Reduce your request rate or batch tokens. You can read the"
        " OpenAI rate limits"
        " here:\nhttps://platform.openai.com/account/rate-limits" + "\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write(
        "- If you are using the free plan, you can upgrade to the paid"
        " plan"
        " here:\nhttps://platform.openai.com/account/billing/overview" + "\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write(
        "- If you are using the paid plan, you can increase your usage"
        " rate limit"
        " here:\nhttps://platform.openai.com/account/billing/limits" + "\n"
    )


def handle_authentication_error():
    """
    Provides guidance on how to handle an authentication error.
    """
    sys.stderr.write(
        f"{RED}Error:{RESET} Your API key or token is invalid, expired, or"
        " revoked. Check your API key or token and make sure it is correct"
        " and active.\n"
    )
    sys.stderr.write(
        "\nYou may need to generate a new API key from your account"
        " dashboard: https://platform.openai.com/account/api-keys\n"
    )
