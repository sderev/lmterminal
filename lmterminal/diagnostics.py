"""Opt-in request timings; never serialize prompts, bodies, headers or credentials."""

import sys
import time


class RequestDiagnostics:
    """One request's monotonic milestones, written to the original stderr."""

    def __init__(self, verbosity, *, clock=None, stderr=None):
        self.verbosity = verbosity
        self.clock = clock or time.monotonic
        # Capture before Rich Live can redirect stderr through its stdout proxy.
        self.stderr = stderr if stderr is not None else sys.stderr
        self.started = self.clock()
        self.times = {}

    def mark(self, event, *, level=1, detail=""):
        now = self.clock()
        self.times[event] = now
        if self.verbosity >= level:
            suffix = f" {detail}" if detail else ""
            self.stderr.write(f"[lmt +{now - self.started:.3f}s] {event}{suffix}\n")
            self.stderr.flush()

    def first(self, event, *, level=1, detail=""):
        if event not in self.times:
            self.mark(event, level=level, detail=detail)

    def request_prepared(self, request, stream):
        self.mark(
            "request prepared",
            detail=f"model={request.model!r} stream={str(stream).lower()} route=/chat/completions",
        )

    def received(self, response, *, stream):
        model = getattr(response, "model", None)
        detail = f"model={model!r}" if isinstance(model, str) else ""
        self.mark("stream ready" if stream else "response received", level=2, detail=detail)

    def completed(self, *, events, text_chunks, usage):
        self.mark("request complete")
        if self.verbosity < 3:
            return
        counts = []
        # Only numeric usage totals, never the raw SDK object or nested data.
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, field, None)
            if type(value) is int:
                counts.append(f"{field}={value}")
        self.mark(
            "response counts",
            level=3,
            detail=f"events={events} text_chunks={text_chunks} "
            + (" ".join(counts) if counts else "usage=unavailable"),
        )

    def output_complete(self):
        dispatch = self.times.get("request dispatched")
        fields = []
        if dispatch is not None:
            complete = self.times.get("request complete")
            if complete is not None:
                fields.append(f"request_s={complete - dispatch:.3f}")
            for event, label in (
                ("first text received", "request_ttft_s"),
                ("first text flushed", "first_flush_s"),
                ("first Markdown refresh returned", "first_refresh_s"),
            ):
                observed = self.times.get(event)
                if observed is not None:
                    fields.append(f"{label}={observed - dispatch:.3f}")
                elif event == "first text received":
                    fields.append(f"{label}=unavailable")
            if not any(
                event in self.times
                for event in ("first text flushed", "first Markdown refresh returned")
            ):
                fields.append("first_output_s=unavailable")
        self.mark("output complete", detail=" ".join(fields))
