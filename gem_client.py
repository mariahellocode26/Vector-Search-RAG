"""
src/rag_pipeline/gem_client.py
===============================
Centralized Google Gemini API calls with rate limiting, retries, logging,
and persistent per-day usage tracking written to gem_usage.json (same dir
as this file).

Quota targets (Gemini 2.5 Flash, free tier):
  - 10 RPM        — rolling 60-second window
  - 250,000 TPM   — rolling 60-second window
  - 250 RPD       — resets at midnight Pacific Time (America/Los_Angeles)

Usage tracking:
  - gem_usage.json is created automatically next to this file on first call.
  - Each entry records timestamp (ISO-8601 PT), model, prompt_tokens,
    completion_tokens, total_tokens, and latency_ms.
  - The daily bucket (keyed by PT date) resets automatically at midnight PT
    in sync with Google's own quota reset.
  - Read the log any time:
        from src.rag_pipeline.gem_client import read_usage_log
        print(read_usage_log())

Quick-start:
  Sync call:
        from src.rag_pipeline.gem_client import call_gemini
        result = call_gemini("Say hi!", max_tokens=50)
        print(result.content)

  Async call (with shared rate limiter for concurrent workloads):
        import asyncio
        from src.rag_pipeline.gem_client import call_gemini_async, GeminiRateLimiter
        limiter = GeminiRateLimiter()
        result = asyncio.run(call_gemini_async("Say hi!", max_tokens=50, rate_limiter=limiter))
        print(result.content)

  Connection test (terminal):
        uv run python3 -c "
        from src.rag_pipeline.gem_client import call_gemini
        result = call_gemini('Say hi!', max_tokens=20)
        print(result.content)
        "

Environment:
  GEMINI_API_KEY is loaded from the first .env file found by walking up from
  this file's directory toward the filesystem root — no path configuration
  required. Standard key name: GEMINI_API_KEY=AIza...
"""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import requests
import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Directory containing this source file — used for sibling file resolution.
_HERE = Path(__file__).resolve().parent

# Usage log lives next to this file so it stays with the module.
_USAGE_FILE = _HERE / "gem_usage.json"


# ---------------------------------------------------------------------------
# Model & quota constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"

MAX_RETRIES      = 3
RATE_LIMIT_WAIT  = 60  # seconds — base wait on 429

# Hard quota limits for DEFAULT_MODEL (Gemini 2.5 Flash, free tier).
# Update all three if you change DEFAULT_MODEL.
RPM_LIMIT = 10
TPM_LIMIT = 250_000
RPD_LIMIT = 250

# Conservative safety margins — intentionally below the hard limits.
# Do NOT raise these toward the hard limits; the headroom absorbs timing
# jitter, concurrent callers, and tiktoken undercounting vs Gemini's tokenizer.
RPM_SAFE = RPM_LIMIT - 1           # 9   — 1 request headroom
TPM_SAFE = int(TPM_LIMIT * 0.96)   # 240,000 — 4 % headroom
RPD_SAFE = RPD_LIMIT - 10          # 240 — 10 request headroom

# Google resets daily quotas at midnight Pacific Time.
from zoneinfo import ZoneInfo
PACIFIC = ZoneInfo("America/Los_Angeles")

# Approximate tiktoken encoding (cl100k_base is very close for Gemini).
TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# .env auto-detection
# ---------------------------------------------------------------------------

def _find_dotenv() -> Optional[Path]:
    """
    Walk up from this file's directory until we find a .env file or hit the
    filesystem root. Returns the Path if found, else None.
    """
    current = _HERE
    while True:
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def init():
    """
    Load GEMINI_API_KEY from the nearest .env file found above this module.
    Safe to call multiple times — dotenv skips keys already in the environment.
    """
    from dotenv import load_dotenv
    env_path = _find_dotenv()
    if env_path:
        load_dotenv(env_path)
        logger.debug(f"Loaded .env from {env_path}")
    else:
        logger.warning("No .env file found in any parent directory; "
                       "GEMINI_API_KEY must already be set in the environment.")


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

def _today_pt() -> str:
    """Return today's date string in Pacific Time, e.g. '2026-05-17'."""
    return datetime.now(PACIFIC).strftime("%Y-%m-%d")


def _load_usage() -> dict:
    """Load the usage JSON from disk, returning an empty structure if missing."""
    if _USAGE_FILE.exists():
        try:
            return json.loads(_USAGE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"daily": {}, "entries": []}


def _save_usage(data: dict) -> None:
    try:
        _USAGE_FILE.write_text(json.dumps(data, indent=2))
    except OSError as e:
        logger.warning(f"Could not write usage log to {_USAGE_FILE}: {e}")


def _record_usage_json(result: "GeminiResult") -> None:
    """
    Append one call record to gem_usage.json and update the daily totals.
    The daily bucket key is the PT date string; it resets automatically
    because a new date string is used each day.
    """
    data  = _load_usage()
    today = _today_pt()

    # Initialise today's bucket if this is the first call of the day.
    if today not in data["daily"]:
        data["daily"][today] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    bucket = data["daily"][today]
    bucket["requests"]           += 1
    bucket["prompt_tokens"]      += result.prompt_tokens
    bucket["completion_tokens"]  += result.completion_tokens
    bucket["total_tokens"]       += result.total_tokens

    entry = {
        "timestamp_pt": datetime.now(PACIFIC).isoformat(),
        "model":             result.model,
        "prompt_tokens":     result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens":      result.total_tokens,
        "latency_ms":        round(result.latency_ms, 1),
    }
    data["entries"].append(entry)

    _save_usage(data)
    logger.debug(f"Usage recorded — daily totals for {today}: {bucket}")


def read_usage_log() -> dict:
    """
    Return the full usage log as a dict.

    Structure:
        {
          "daily": {
            "2026-05-17": {
              "requests": 4,
              "prompt_tokens": 320,
              "completion_tokens": 180,
              "total_tokens": 500
            }
          },
          "entries": [
            {
              "timestamp_pt": "2026-05-17T10:23:01.123456-07:00",
              "model": "gemini-2.5-flash",
              "prompt_tokens": 80,
              "completion_tokens": 45,
              "total_tokens": 125,
              "latency_ms": 812.3
            },
            ...
          ]
        }
    """
    return _load_usage()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeminiResult:
    content:           str
    latency_ms:        float
    model:             str
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


# ---------------------------------------------------------------------------
# Rate limiter (async, in-process)
# ---------------------------------------------------------------------------

class GeminiRateLimiter:
    """
    Tracks RPM, TPM, and RPD with sliding windows.

    - RPM & TPM: true rolling 60-second window via a timestamped deque.
      Google uses a rolling window (not a fixed-clock minute), so when the
      window is full the correct wait is until the *oldest* entry expires,
      not a flat 1-second sleep.
    - RPD: calendar day in Pacific Time. Google resets daily quotas at
      midnight PT, not midnight UTC.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        # Each entry: (monotonic_timestamp, tokens_used)
        self._requests: deque[Tuple[float, int]] = deque()
        self._daily_requests: int = 0
        self._daily_reset = datetime.now(PACIFIC).date()

    def _cleanup_windows(self):
        cutoff = time.monotonic() - 60
        while self._requests and self._requests[0][0] < cutoff:
            self._requests.popleft()

    def _check_daily_reset(self):
        today = datetime.now(PACIFIC).date()
        if today > self._daily_reset:
            self._daily_requests = 0
            self._daily_reset = today

    async def wait_if_needed(self, estimated_tokens: int = 0):
        """
        Block until it is safe to make another request.

        Sleeps exactly as long as needed for the oldest window entry to expire
        rather than a flat fixed duration — correct behavior for a sliding window.
        """
        async with self._lock:
            self._cleanup_windows()
            self._check_daily_reset()

            current_rpm = len(self._requests)
            current_tpm = sum(tokens for _, tokens in self._requests)

            if (
                current_rpm >= RPM_SAFE
                or current_tpm + estimated_tokens > TPM_SAFE
                or self._daily_requests >= RPD_SAFE
            ):
                if self._requests:
                    oldest_ts = self._requests[0][0]
                    wait_time = max(0.1, (oldest_ts + 60) - time.monotonic())
                else:
                    wait_time = 1.0

                logger.warning(
                    f"GeminiRateLimiter: approaching limits "
                    f"(RPM:{current_rpm}/{RPM_LIMIT}, "
                    f"TPM:~{current_tpm}/{TPM_LIMIT}, "
                    f"RPD:{self._daily_requests}/{RPD_LIMIT}). "
                    f"Waiting {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

    async def record_usage(self, tokens: int):
        """Record a confirmed successful request."""
        async with self._lock:
            self._cleanup_windows()
            self._check_daily_reset()
            self._requests.append((time.monotonic(), tokens))
            self._daily_requests += 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(TOKEN_ENCODER.encode(text))


def _is_rate_limit_error(resp_or_exc) -> bool:
    if isinstance(resp_or_exc, Exception):
        msg = str(resp_or_exc).lower()
    else:
        msg = (
            str(getattr(resp_or_exc, "text", ""))
            + str(getattr(resp_or_exc, "status_code", ""))
        )
    return any(k in msg for k in ("429", "quota", "rate limit", "resource exhausted"))


def _build_payload(prompt: str, max_tokens: int, temperature: float = 0.3) -> dict:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_gemini(
    prompt: str,
    max_tokens: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> GeminiResult:
    """
    Synchronous Gemini call with retries and usage logging.

    For high-throughput concurrent workloads use call_gemini_async with a
    shared GeminiRateLimiter instead.

    Args:
        prompt:      The user prompt to send.
        max_tokens:  Maximum tokens in the completion (required).
        model:       Gemini model string. Defaults to DEFAULT_MODEL.
        temperature: Sampling temperature. Defaults to 0.3.

    Returns:
        GeminiResult with .content, .latency_ms, and token counts.
    """
    init()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file as:\n"
            "    GEMINI_API_KEY=AIza..."
        )

    estimated_prompt_tokens = _count_tokens(prompt)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent"
    )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.monotonic()

            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": api_key},
                json=_build_payload(prompt, max_tokens, temperature),
                timeout=120,
            )

            elapsed = (time.monotonic() - t0) * 1000

            if resp.status_code != 200:
                if _is_rate_limit_error(resp):
                    wait = RATE_LIMIT_WAIT * (attempt + 1)
                    logger.warning(
                        f"Rate limit hit (HTTP {resp.status_code}), waiting {wait}s"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            data    = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            completion_tokens = _count_tokens(content) if content else 0

            result = GeminiResult(
                content=content.strip(),
                latency_ms=elapsed,
                model=model,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=estimated_prompt_tokens + completion_tokens,
            )

            _record_usage_json(result)
            return result

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Gemini call failed after {MAX_RETRIES} attempts: {e}")
                raise

            wait = RATE_LIMIT_WAIT * (attempt + 1) if _is_rate_limit_error(e) else 5.0
            logger.warning(
                f"Gemini error (attempt {attempt + 1}): {e} — waiting {wait:.0f}s"
            )
            time.sleep(wait)


async def call_gemini_async(
    prompt: str,
    max_tokens: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    rate_limiter: Optional[GeminiRateLimiter] = None,
) -> GeminiResult:
    """
    Async Gemini call with sliding-window rate limiting and usage logging.

    Pass a shared GeminiRateLimiter so all concurrent callers draw from the
    same RPM/TPM/RPD budget.

    Args:
        prompt:       The user prompt to send.
        max_tokens:   Maximum tokens in the completion (required).
        model:        Gemini model string. Defaults to DEFAULT_MODEL.
        temperature:  Sampling temperature. Defaults to 0.3.
        rate_limiter: Optional shared GeminiRateLimiter instance.

    Returns:
        GeminiResult with .content, .latency_ms, and token counts.
    """
    init()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file as:\n"
            "    GEMINI_API_KEY=AIza..."
        )

    estimated_prompt_tokens = _count_tokens(prompt)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent"
    )

    if rate_limiter:
        await rate_limiter.wait_if_needed(estimated_prompt_tokens)

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.monotonic()

            import aiohttp
            async with asyncio.timeout(120):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        params={"key": api_key},
                        json=_build_payload(prompt, max_tokens, temperature),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            if _is_rate_limit_error(resp) or _is_rate_limit_error(text):
                                if rate_limiter:
                                    await rate_limiter.wait_if_needed()
                                raise Exception(f"Rate limit: {text}")
                            resp.raise_for_status()
                        data = await resp.json()

            elapsed = (time.monotonic() - t0) * 1000
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            completion_tokens = _count_tokens(content) if content else 0

            result = GeminiResult(
                content=content.strip(),
                latency_ms=elapsed,
                model=model,
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=estimated_prompt_tokens + completion_tokens,
            )

            if rate_limiter:
                await rate_limiter.record_usage(result.total_tokens)

            _record_usage_json(result)
            return result

        except Exception as e:
            is_rl = _is_rate_limit_error(e)
            if is_rl and rate_limiter:
                await rate_limiter.wait_if_needed()

            if attempt == MAX_RETRIES - 1:
                logger.error(f"Gemini async failed after {MAX_RETRIES} attempts: {e}")
                raise

            wait = RATE_LIMIT_WAIT * (attempt + 1) if is_rl else 5.0
            logger.warning(
                f"Gemini async error (attempt {attempt + 1}): {e} — sleeping {wait:.1f}s"
            )
            await asyncio.sleep(wait)