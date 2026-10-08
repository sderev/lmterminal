"""Chat Completions transport; callers own clients and presentation."""

import time


def send_prepared_request(
    client, request, *, n=1, stop=None, stream=False, on_text=None, diagnostics=None
):
    """Return (text, elapsed seconds, raw response or collected stream events).

    No output or client discovery occurs here. Nonempty streaming text is delivered
    synchronously to on_text before requesting the next event. Errors propagate;
    the caller owns client lifetime. Text is never newline-normalized.
    """
    start_time = time.monotonic_ns()
    request_kwargs = dict(
        messages=request.messages, model=request.model, n=n, stream=stream, **request.controls
    )
    if stop is not None:
        request_kwargs["stop"] = stop
    if diagnostics:
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

            if on_text is not None and delta.content:
                on_text(delta.content)

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
