import sys
import time
from dataclasses import dataclass

import openai
import tiktoken

from .model_registry import get_input_price_per_million, get_price_band, get_tokenizer_model

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
    reasoning_effort=None,
    request_options=None,
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
    if stop is not None:
        request_kwargs["stop"] = stop
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if request_options:
        request_kwargs.update(request_options)

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
    model = get_tokenizer_model(model)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model=DEFAULT_MODEL):
    """Returns the number of tokens used by a list of messages."""
    model = get_tokenizer_model(model)
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


@dataclass(frozen=True)
class PromptCostEstimate:
    num_tokens: int
    price_per_1m_tokens: float
    cost: str
    pricing_context: str | None


def estimate_prompt_cost_details(message, model):
    """Returns prompt token and pricing details for a model."""
    num_tokens = num_tokens_from_messages(message, model)
    price_per_1m_tokens = get_input_price_per_million(model, num_tokens)
    _, pricing_context = get_price_band(model, num_tokens)

    return PromptCostEstimate(
        num_tokens=num_tokens,
        price_per_1m_tokens=price_per_1m_tokens,
        cost=estimated_cost(num_tokens, price_per_1m_tokens),
        pricing_context=pricing_context,
    )


def estimate_prompt_cost(message, model):
    """Returns the estimated cost of a prompt."""
    return estimate_prompt_cost_details(message, model).cost


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
