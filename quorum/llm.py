"""Thin wrapper around the Claude API: one structured-JSON call, with token and cost accounting."""

import json
import os
import threading
from dataclasses import dataclass, field

import anthropic

from quorum import config

# USD per million tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


class AgentError(RuntimeError):
    """The model could not produce a usable answer (refusal, truncation, bad JSON, API failure)."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, inp: int, out: int) -> None:
        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out
            self.calls += 1

    def cost(self, model: str) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


class LLM:
    def __init__(self, model: str = config.MODEL):
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise AgentError("ANTHROPIC_API_KEY is not set. Use --mock for an offline demo run.")
        self.model = model
        self.client = anthropic.Anthropic(timeout=180.0, max_retries=3)
        self.usage = Usage()

    def json(self, system: str, prompt: str, schema: dict, effort: str = "low", max_tokens: int = 16000) -> dict:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise AgentError("The Anthropic API key was rejected.") from e
        except anthropic.APIError as e:
            raise AgentError(f"API request failed: {e}") from e
        self.usage.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise AgentError(f"Model stopped early ({response.stop_reason}).")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise AgentError("Model returned invalid JSON.") from e
