"""Claude API wrapper: structured-JSON calls, run singly or as a Message Batch, with exact cost accounting.

The daily job is unattended, so by default its requests go through the Message Batches API at half price. A batch
that hasn't finished within QUORUM_BATCH_WAIT_MIN minutes is cancelled, and whatever it didn't finish runs as normal
requests, so a slow batch can delay the site but never lose the day. A daily spending cap (QUORUM_MAX_DAILY_USD)
stops further calls once reached.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

from quorum import config

# USD per million tokens, from the Pricing page (platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-13):
# (input, output, cache-read multiplier on input). Cache writes (5-minute) bill 1.25x input; batches bill 0.5x of all.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.05),
    "claude-sonnet-5-5": (2.0, 10.0, 0.05),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
    "claude-fable-5-1": (10.0, 50.0, 0.025),
}
CACHE_WRITE = 1.25
BATCH_DISCOUNT = 0.5


class AgentError(RuntimeError):
    """The model could not produce a usable answer (refusal, truncation, bad JSON, API failure, budget)."""


@dataclass
class Job:
    """One structured-output request. `id` must be unique in a run and match [A-Za-z0-9_-]{1,64}."""

    id: str
    system: str
    prompt: str
    schema: dict
    effort: str = "low"
    max_tokens: int = 8000


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    batched_calls: int = 0
    _cost: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage, batched: bool) -> None:
        pin, pout, read_mult = PRICES.get(model, (0.0, 0.0, 0.1))
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        dollars = (inp * pin + cw * pin * CACHE_WRITE + cr * pin * read_mult + out * pout) / 1e6
        with self._lock:
            self.calls += 1
            self.batched_calls += int(batched)
            self.input_tokens += inp
            self.output_tokens += out
            self.cache_write_tokens += cw
            self.cache_read_tokens += cr
            self._cost += dollars * (BATCH_DISCOUNT if batched else 1.0)

    def cost(self, model: str | None = None) -> float:
        return self._cost


def _parse(message) -> dict:
    if message.stop_reason in ("refusal", "max_tokens"):
        raise AgentError(f"Model stopped early ({message.stop_reason}).")
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise AgentError("Model returned invalid JSON.") from e


class LLM:
    def __init__(self, model: str = config.MODEL, client=None):
        if client is None and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise AgentError("ANTHROPIC_API_KEY is not set. Use --mock for an offline demo run.")
        self.model = model
        self.client = client or anthropic.Anthropic(timeout=300.0, max_retries=3)
        self.usage = Usage()
        self.batch = config.USE_BATCH
        self.batch_wait = config.BATCH_WAIT_MIN * 60
        self.poll_seconds = 20
        self.budget = config.MAX_DAILY_USD

    # ---------- single requests ----------

    def _params(self, job: Job) -> dict:
        return {
            "model": self.model,
            "max_tokens": job.max_tokens,
            "system": job.system,
            "messages": [{"role": "user", "content": job.prompt}],
            "output_config": {"effort": job.effort, "format": {"type": "json_schema", "schema": job.schema}},
        }

    def _check_budget(self) -> None:
        if self.budget and self.usage.cost() >= self.budget:
            raise AgentError(f"Daily budget of ${self.budget:.2f} reached; skipping further calls.")

    def call(self, job: Job) -> dict:
        self._check_budget()
        try:
            response = self.client.beta.messages.create(
                **self._params(job),
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # a policy decline is retried on a fallback model in the same call
            )
        except anthropic.AuthenticationError as e:
            raise AgentError("The Anthropic API key was rejected.") from e
        except anthropic.APIError as e:
            raise AgentError(f"API request failed: {e}") from e
        self.usage.add(self.model, response.usage, batched=False)
        return _parse(response)

    def json(self, system: str, prompt: str, schema: dict, effort: str = "low", max_tokens: int = 8000) -> dict:
        return self.call(Job("single", system, prompt, schema, effort, max_tokens))

    # ---------- many requests ----------

    def run_many(self, jobs: list[Job]) -> dict[str, dict | AgentError]:
        """Run independent jobs, batched when enabled. Every job id gets a result or an AgentError."""
        if not jobs:
            return {}
        try:
            self._check_budget()
        except AgentError as e:
            return {j.id: e for j in jobs}
        done: dict[str, dict | AgentError] = {}
        if self.batch:
            try:
                done = self._run_batch(jobs)
            except anthropic.APIError as e:
                print(f"  batch request failed ({e}); running these {len(jobs)} requests directly")
        rest = [j for j in jobs if j.id not in done]
        if rest:
            done.update(self._run_direct(rest))
        return done

    def _run_direct(self, jobs: list[Job]) -> dict[str, dict | AgentError]:
        def one(job):
            try:
                return job.id, self.call(job)
            except AgentError as e:
                return job.id, e

        with ThreadPoolExecutor(max_workers=6) as pool:
            return dict(pool.map(one, jobs))

    def _run_batch(self, jobs: list[Job]) -> dict[str, dict | AgentError]:
        """Submit, wait, collect. Unfinished or retryable requests are left out so the caller runs them directly."""
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": j.id, "params": self._params(j)} for j in jobs]
        )
        print(f"  batch {batch.id}: {len(jobs)} requests submitted")
        deadline = time.time() + self.batch_wait
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.time() > deadline:
                print(f"  batch {batch.id} still running after {self.batch_wait / 60:.0f} min; cancelling")
                self.client.messages.batches.cancel(batch.id)
                for _ in range(30):  # cancellation finishes quickly; finished requests keep their results
                    time.sleep(min(5, self.poll_seconds))
                    batch = self.client.messages.batches.retrieve(batch.id)
                    if batch.processing_status == "ended":
                        break
                break
            time.sleep(self.poll_seconds)
        out: dict[str, dict | AgentError] = {}
        if batch.processing_status != "ended":
            return out
        for r in self.client.messages.batches.results(batch.id):
            kind = r.result.type
            if kind == "succeeded":
                self.usage.add(self.model, r.result.message.usage, batched=True)
                try:
                    out[r.custom_id] = _parse(r.result.message)
                except AgentError as e:
                    out[r.custom_id] = e
            elif kind == "errored":
                err = r.result.error
                etype = getattr(getattr(err, "error", None), "type", None) or getattr(err, "type", None)
                if etype in ("invalid_request", "invalid_request_error"):
                    out[r.custom_id] = AgentError(f"Request rejected: {etype}")
                # other errors (server, overloaded) are retried directly by the caller
            # canceled / expired: left for a direct retry
        return out
