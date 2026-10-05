import sys
import time

import openai

from .request_options import DEFAULT_MODEL as DEFAULT_MODEL  # noqa: PLC0414 -- public constant
from .request_options import UNSET, prepare_request

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
    model=UNSET,
    # max_tokens=3900,
    n=1,
    temperature=UNSET,
    stop=None,
    stream=False,
    update_markdown_stream=None,
    *,
    reasoning_effort=UNSET,
    request_options=None,
    diagnostics=None,
):
    """
    Sends a request to the OpenAI Chat API.

    Omitted model/effort use Luna/none; explicit None omits the control.

    Returns:
        tuple[str, float, object]:
            * generated_text
            * response_time (seconds)
            * raw_response (non-stream) or collected stream chunks (stream)
    """
    request = prepare_request(model, prompt, temperature, reasoning_effort, request_options)
    if diagnostics:
        diagnostics.request_prepared(request, stream)
    return send_prepared_request(
        api_key,
        request,
        n,
        stop,
        stream,
        update_markdown_stream,
        diagnostics=diagnostics,
    )


def send_prepared_request(
    api_key, request, n=1, stop=None, stream=False, update_markdown_stream=None, *, diagnostics=None
):
    """Send a finalized request without recomposing or revalidating it."""
    start_time = time.monotonic_ns()
    request_kwargs = dict(
        messages=request.messages, model=request.model, n=n, stream=stream, **request.controls
    )
    if stop is not None:
        request_kwargs["stop"] = stop
    client = _get_client(api_key)
    if diagnostics:
        diagnostics.mark("client ready", level=2)
        diagnostics.mark("request dispatched")
    response = client.chat.completions.create(**request_kwargs)
    if diagnostics:
        diagnostics.received(response, stream=stream)
    usage = getattr(response, "usage", None)
    events = text_chunks = 0

    if stream:
        # Create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []

        # Iterate through the stream of events
        for chunk in response:
            events += 1
            if diagnostics:
                model = getattr(chunk, "model", None)
                diagnostics.first(
                    "first stream event received",
                    level=2,
                    detail=f"model={model!r}" if isinstance(model, str) else "",
                )
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            collected_chunks.append(chunk)  # save the event response
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta  # extract the delta
            if delta.content:
                text_chunks += 1
                if diagnostics:
                    diagnostics.first("first text received")
            if delta.content is not None:
                collected_messages.append(delta.content)  # save the message

            if update_markdown_stream:
                update_markdown_stream(delta.content or "")
            else:
                if diagnostics and delta.content:
                    diagnostics.first("first text submitted to output", level=2)
                print(delta.content or "", end="", flush=True)
                if diagnostics and delta.content:
                    diagnostics.first("first text flushed")

        # Save the time delay and text received
        response_time = (time.monotonic_ns() - start_time) / 1e9
        generated_text = "".join(collected_messages)
        response_payload = collected_chunks

    else:
        # Extract and save the generated response
        generated_text = response.choices[0].message.content or ""
        if diagnostics and generated_text:
            diagnostics.first("first text received")

        # Save the time delay
        response_time = (time.monotonic_ns() - start_time) / 1e9
        response_payload = response

    if diagnostics:
        diagnostics.completed(events=events, text_chunks=text_chunks, usage=usage)
    return (
        generated_text,
        response_time,
        response_payload,
    )


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
