#!/usr/bin/env python3
"""
deepship.py
-----------
Pipeline + live GUI that, for a list of college domains (e.g. princeton.edu):

  1. CRAWL   — fetches the college home page and pulls out every link on it
               (via BeautifulSoup), same style as alting_crawler.py's link
               scraping.
  2. TRIAGE  — sends that link list (url + anchor text) to Groq, same
               call_groq() shape as alting_ua.py, and asks it which of the
               links are most likely the financial aid / scholarships page.
               Groq replies with a JSON list of candidate URLs + a reason.
  3. FETCH   — for each candidate link, fetches the page and uses
               BeautifulSoup to pull out its visible text (scripts/styles
               stripped), same extract_page_text() approach as alting_ua.py.
  4. EXTRACT (two passes) —
       Pass 1 (loose sweep): splits the page's visible text into chunks and
       asks Groq, per chunk, to just pull out any raw bits of text that
       plausibly mention a scholarship, need-based aid, or full-ride/
       full-tuition award -- no structuring, no filtering, cast a wide net.
       Pass 2 (structuring): takes everything pass 1 found for that page and
       sends it back to Groq to be deduped, merged, and turned into tighter,
       structured records (name, eligibility, amount, deadline, how to
       apply, etc). Both passes' output are kept and shown -- pass 1's raw
       snippets and pass 2's structured entries.
       Pass 3 (FAQ gap-fill): if Groq flagged a FAQ page during triage,
       fetches it and walks it through Groq to enrich the pass 2 entries
       with anything the FAQ adds.
       Pass 4 (regex cross-check, no Groq call): purely mechanical. Re-scans
       the *raw* BeautifulSoup text already pulled for every finaid page and
       the FAQ page for whole paragraphs containing any of a fixed list of
       financial-aid keywords (financial aid, need-based, need-blind,
       scholarship, grant, fellowship, FAFSA, work-study, etc). Every
       matching paragraph is kept in full, then checked for word-overlap
       against pass 3's (or pass 2's, if no FAQ) structured output so
       paragraphs that don't seem reflected anywhere in the LLM output get
       flagged as possibly-missed / new info for a human to double check.

Adaptive chunking (the "clump to 4, clump to 6" behavior):
  A page is first split in TWO and each half is sent to Groq. If Groq (or
  the HTTP layer in front of it) comes back with a 413 / "payload too
  large" style error, that page is retried from scratch split into FOUR
  chunks instead. If that *still* 413s, it's retried again split into SIX
  chunks — the ceiling. If that also fails, the page falls back to a
  single hard-truncated chunk so the run never just dies on one page.

  Whenever a page is chunked (2/4/6), each chunk after the first carries a
  short JSON summary of what the *previous* chunk's Groq call extracted,
  fed back in as context in the next prompt, so a scholarship entry that
  gets physically cut in half by the chunk boundary doesn't confuse the
  model into either duplicating it or losing it.

GUI:
  Running this script with no arguments launches a dark-theme Tkinter GUI
  (same visual language as alting_crawler.py) that shows the whole process
  live, split into four panes:
      - Crawled Site           (home page + every link found on it)
      - Extracted Links/Pages  (the candidate finaid links Groq picked,
                                 with which one is currently being deep-dived)
      - Extracted Text         (the BeautifulSoup visible-text pulled from
                                 whichever page is currently being processed)
      - Groq's Response        (a live feed of every raw Groq reply, plus a
                                 running table of every scholarship found)
  A stats bar tracks colleges done/total, links found, candidates, pages
  fetched, scholarships found, and Groq calls made, live, as the crawl runs.
  You can paste in any number of college domains and pick exactly how many
  of them to actually run this pass with a "Fetch first N" spinbox.

  Nothing is written to disk by the GUI — results are for on-screen review
  only, as requested. The CLI path (run with arguments) still exists and
  still writes a JSON file, for scripted/headless use.

Requirements:
    pip install requests beautifulsoup4

Config (env vars, or a .env file sitting next to this script):
    OLLAMA_BASE_URL=https://offr.tail05ae98.ts.net   (your local/Tailscale Ollama
                                                        server; no API key needed)
    OLLAMA_MODEL=qwen2.5:1.5b                         (optional -- overrides the
                                                        default model below)

    Every LLM call in this script (link triage, pass 1/2/3) goes through
    your local Ollama server -- no cloud provider, no API key required.
    Use the GUI's "Check API Key" button, or the CLI's startup check, to
    confirm the server/model that's configured.

Usage:
    python3 deepship.py                          # launches the GUI
    python3 deepship.py princeton.edu harvard.edu    # CLI, writes JSON
    python3 deepship.py --file colleges.txt --output finaid_results.json
"""

import argparse
import json
import os
import re
import sys
import time
import threading
import queue
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# .env loader (same behavior as alting_ua.py: KEY=VALUE lines, no deps,
# doesn't override anything already set in the real environment)
# ---------------------------------------------------------------------------

def _load_dotenv_if_present():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    info = {"env_path": env_path, "file_found": False, "keys_loaded": []}
    if not os.path.exists(env_path):
        return info
    info["file_found"] = True
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                info["keys_loaded"].append(key)
    return info


ENV_DEBUG = _load_dotenv_if_present()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://offr.tail05ae98.ts.net").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_ENDPOINT = f"{OLLAMA_BASE_URL}/v1/chat/completions"


def _refresh_groq_env():
    """Re-runs the .env loader and re-reads OLLAMA_BASE_URL / OLLAMA_MODEL
    from os.environ. Safe to call repeatedly (e.g. right before Start is
    clicked in the GUI) so editing .env or exporting the var AFTER the
    script/GUI process already started doesn't require a full restart --
    _load_dotenv_if_present() only fills in keys not already in
    os.environ, so this never clobbers something already set. (Name kept
    for backwards compatibility with the rest of the GUI's call sites.)"""
    global ENV_DEBUG, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_ENDPOINT, GROQ_TIMEOUT_S
    ENV_DEBUG = _load_dotenv_if_present()
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    OLLAMA_ENDPOINT = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    GROQ_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", GROQ_TIMEOUT_S))


def groq_key_diagnostic():
    """Builds a status string about the local Ollama server config -- no
    API key involved, just which base URL and model are configured and
    where OLLAMA_BASE_URL/OLLAMA_MODEL came from. Kept the old function
    name for backwards compatibility with existing call sites."""
    lines = [f"Ollama server: {OLLAMA_BASE_URL}", f"Model: {OLLAMA_MODEL}", "No API key required."]
    source = "a .env file" if "OLLAMA_BASE_URL" in ENV_DEBUG.get("keys_loaded", []) else "the default / shell environment"
    lines.append(f"Base URL loaded from: {source}.")
    return "\n".join(lines)

USER_AGENT = "DeepshipFinAidCrawler/1.0 (college financial aid research; contact: set-your-email-here)"
REQUEST_TIMEOUT_S = 25
# A 60s ceiling is fine for short chat turns, but link-triage prompts can run
# 10-15k+ chars (a linky home page), and a small local model on CPU can take
# well over a minute to chew through that -- especially on the very first
# call, which also pays the one-time cost of the model loading into memory.
# Override with OLLAMA_TIMEOUT_S if even 180s isn't enough on your hardware.
GROQ_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "180"))
POLITE_DELAY_S = 0.75

# The chunking ladder described above: try whole-page-in-2, then 4, then 6.
SPLIT_LADDER = [2, 4, 6]

DEFAULT_MAX_CANDIDATES = 3
MAX_LINKS_SENT_TO_GROQ = 250  # trim an absurdly linky home page before triage

# ---------------------------------------------------------------------------
# Manual-paste mode: instead of actually POSTing to an LLM API, every call
# pops up a window with the full prompt + a Copy button, and blocks until
# you paste the model's reply back in and hit Submit. On by default; set
# MANUAL_PASTE_MODE=0 in the environment / .env to go back to real HTTP
# calls against the providers configured below.
# ---------------------------------------------------------------------------
MANUAL_PASTE_MODE = os.environ.get("MANUAL_PASTE_MODE", "1").strip() not in ("0", "false", "False", "")

# Set by the GUI's App.__init__ so manual-paste popups triggered from the
# background crawler thread can be safely built on the main (Tk) thread
# instead. Stays None in CLI mode, where call_groq() runs on the main
# thread anyway.
_GUI_ROOT = None

DEFAULT_SAMPLE_COLLEGES = [
    "princeton.edu", "harvard.edu", "mit.edu", "yale.edu", "stanford.edu",
    "williams.edu", "amherst.edu", "pomona.edu", "swarthmore.edu", "duke.edu",
]


# ---------------------------------------------------------------------------
# Shared Groq rate limiter (same FIFO/adaptive-backoff design as
# splicer_core.py's GroqRateLimiter, trimmed down)
# ---------------------------------------------------------------------------

GROQ_MIN_INTERVAL_SECONDS = 3.0
GROQ_MAX_INTERVAL_SECONDS = 60.0
GROQ_429_BACKOFF_STEP = 4.0
GROQ_COOLDOWN_STEP = 1.0


class GroqRateLimiter:
    def __init__(self, min_interval=GROQ_MIN_INTERVAL_SECONDS, max_interval=GROQ_MAX_INTERVAL_SECONDS):
        self.base_interval = min_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._consecutive_429 = 0

    def wait_turn(self, log_fn=None):
        with self._lock:
            now = time.time()
            wait = self._next_allowed_at - now
            if wait > 0:
                if log_fn:
                    log_fn(f"  (throttling Groq calls to 1/{self.min_interval:.0f}s -- waiting {wait:.1f}s)")
                time.sleep(wait)
            self._next_allowed_at = time.time() + self.min_interval

    def note_rate_limited(self, log_fn=None):
        with self._lock:
            self._consecutive_429 += 1
            old = self.min_interval
            self.min_interval = min(self.max_interval, self.min_interval + GROQ_429_BACKOFF_STEP * self._consecutive_429)
            self._next_allowed_at = max(self._next_allowed_at, time.time() + self.min_interval)
            if log_fn and self.min_interval != old:
                log_fn(f"  (Groq 429 -- raising throttle interval {old:.0f}s -> {self.min_interval:.0f}s)")

    def note_success(self):
        with self._lock:
            self._consecutive_429 = 0
            if self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval - GROQ_COOLDOWN_STEP)


GROQ_RATE_LIMITER = GroqRateLimiter()  # legacy name kept in case anything external
                                        # imports it; the pipeline itself now uses a
                                        # separate GroqRateLimiter per provider (see
                                        # _rate_limiter_for() below)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PayloadTooLargeError(Exception):
    """Raised when a provider (or a proxy in front of it) rejects a request
    for being too big -- HTTP 413, or a 400 whose body says as much (some
    providers report oversized-request as a 400 with a message instead of
    a clean 413). NOT provider-availability related, so this always bubbles
    straight up to the caller (which re-chunks via SPLIT_LADDER) instead of
    triggering a provider fallback -- a smaller chunk on the SAME provider
    is the fix, not a different provider."""
    pass


class GroqCallError(Exception):
    """Base class for "this provider call didn't work" errors. Kept this
    name for backwards compatibility with the rest of the pipeline, which
    still catches GroqCallError everywhere -- QuotaExhaustedError and
    ProviderAuthError below are both subclasses so existing except clauses
    keep working unchanged."""
    pass


class QuotaExhaustedError(GroqCallError):
    """A provider's retries all came back 429 -- reads as "tapped out"
    (free-tier daily/rate quota) rather than a one-off blip. Triggers an
    automatic fallback to the next configured provider."""
    pass


class ProviderAuthError(GroqCallError):
    """A provider rejected the API key outright (401/403) -- missing,
    invalid, or revoked. Also triggers fallback to the next provider,
    since retrying the same bad key won't help."""
    pass


class StopRequested(Exception):
    """Raised internally to unwind out of the pipeline when the GUI's Stop
    button (or a stop_flag threading.Event) is set mid-run."""
    pass


def _check_stop(stop_flag):
    if stop_flag is not None and stop_flag.is_set():
        raise StopRequested()


# ---------------------------------------------------------------------------
# LLM config -- local Ollama server only, no cloud provider, no API key
# ---------------------------------------------------------------------------
# LLM_PROVIDERS is still a list (the rest of the pipeline's fallback-chain
# machinery is written generically against it), but it now holds exactly
# one entry: your local Ollama server, reached over Tailscale, speaking
# its OpenAI-compatible POST /v1/chat/completions endpoint. No API key is
# read or required -- api_key_env is None for this entry.

def _build_llm_providers():
    """Builds the provider list fresh from the CURRENT OLLAMA_BASE_URL /
    OLLAMA_MODEL globals every time it's called, instead of freezing them
    at import time. This is what makes _refresh_groq_env() actually work
    as documented -- without this, a .env edit or a different host picked
    up mid-session would update the globals but every real HTTP call
    would keep silently hitting the stale endpoint/model baked into a
    module-load-time list."""
    return [
        {
            # Local Ollama server (reached over Tailscale), speaking Ollama's
            # OpenAI-compatible /v1/chat/completions endpoint. No API key --
            # api_key_env is None, which the code below treats as "always
            # configured, no Authorization header needed."
            "name": "ollama",
            "endpoint": f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "api_key_env": None,
            "model_env": "OLLAMA_MODEL",
            "models": [OLLAMA_MODEL],
        },
    ]


# Kept as a plain name for any code that still refers to LLM_PROVIDERS
# directly -- but this is a *snapshot*, not the source of truth. Every
# call site that matters (_configured_providers, llm_provider_diagnostic)
# calls _build_llm_providers() itself so it always sees the live config.
LLM_PROVIDERS = _build_llm_providers()

_PROVIDER_RATE_LIMITERS = {}
_PROVIDER_STATE_LOCK = threading.Lock()
_ACTIVE_PROVIDERS = None        # lazily built; providers with a key actually set
_DEAD_COMBOS = set()            # {(provider_name, model), ...} that failed hard this run
_LAST_GOOD_COMBO = None         # sticky (provider_name, model) tried first, so we
                                 # don't re-probe a dead combo on every single call


def _rate_limiter_for(provider_name, model):
    # Rate limits on most of these providers (Groq included) are tracked
    # per MODEL, not just per account -- so each (provider, model) pair
    # gets its own independent throttle/backoff state, not one shared per
    # provider. That also means one model on a provider tapping out its
    # quota doesn't slow down a different model on that same provider.
    key = f"{provider_name}::{model}"
    if key not in _PROVIDER_RATE_LIMITERS:
        _PROVIDER_RATE_LIMITERS[key] = GroqRateLimiter()
    return _PROVIDER_RATE_LIMITERS[key]


def _models_for(provider):
    """This provider's model list, with its <PROVIDER>_MODEL env override
    (if set) pulled to the front -- so e.g. GROQ_MODEL=llama-3.3-70b-versatile
    still works exactly like a manual pin, it's just now the FIRST thing
    tried on that provider rather than the ONLY thing."""
    models = list(provider["models"])
    override = os.environ.get(provider["model_env"], "").strip()
    if override:
        models = [override] + [m for m in models if m != override]
    return models


def _build_candidate_chain():
    """Flattens every configured provider's model list into one ordered
    list of (provider, model) candidates: all of provider 1's models
    (its override first, then its list in order), then all of provider
    2's, etc. -- so a rate-limited/decommissioned MODEL falls back to the
    next model on the SAME provider before ever moving to a different
    provider."""
    chain = []
    for p in _configured_providers():
        for m in _models_for(p):
            chain.append((p, m))
    return chain


def _configured_providers():
    """Returns LLM_PROVIDERS filtered down to the ones that are usable
    right now: providers with api_key_env=None (local/no-auth, like our
    Ollama server) are always considered configured; any others still
    need a real key present in the environment."""
    global _ACTIVE_PROVIDERS
    active = []
    for p in _build_llm_providers():
        if p["api_key_env"] is None:
            active.append(p)
            continue
        key = os.environ.get(p["api_key_env"], "").strip()
        if key and key not in ("PASTE_YOUR_GROQ_API_KEY_HERE", "paste"):
            active.append(p)
    _ACTIVE_PROVIDERS = active
    return active


_PASTED_KEY_ENV_VAR_NAMES = {p["api_key_env"] for p in LLM_PROVIDERS if p["api_key_env"] is not None}


def _detect_provider_for_raw_key(key):
    """No-op now that the only configured provider is the local Ollama
    server (no API key needed). Kept as a stub so any leftover call
    sites don't blow up; always returns None."""
    return None


def _assign_pasted_keys(raw_text):
    """No-op now that the only configured provider is the local Ollama
    server (no API key needed). Kept as a stub, with the same
    (assigned, unrecognized) return shape, so any leftover call sites
    don't blow up."""
    return [], []


def reset_provider_fallback_state():
    """Call this at the start of a fresh run (CLI main(), GUI Start
    button) so any (provider, model) combo marked dead on a previous run
    gets a clean shot again -- useful if a daily quota reset, or you've
    added/rotated a key mid-session."""
    global _DEAD_COMBOS, _LAST_GOOD_COMBO
    _DEAD_COMBOS = set()
    _LAST_GOOD_COMBO = None


def llm_provider_diagnostic():
    """Human-readable status of every provider AND every model on it --
    which env var the key comes from, whether a key is set, each model's
    place in that provider's fallback order, and (mid-run) whether a given
    (provider, model) combo is currently marked dead. Used by the GUI's
    "Check API Key" button and the CLI's startup check, and folded into
    the error raised when every combo in the whole chain has failed."""
    lines = ["LLM provider + model fallback chain (tried in this order):"]
    for p in _build_llm_providers():
        if p["api_key_env"] is None:
            lines.append(f"  - {p['name']} [{p['endpoint']}]: local server, no API key needed")
            for m in _models_for(p):
                dead_note = "  -- marked DEAD this run" if (p["name"], m) in _DEAD_COMBOS else ""
                lines.append(f"      · {m}{dead_note}")
            continue
        key = os.environ.get(p["api_key_env"], "").strip()
        if key:
            masked = key[:4] + "…" + key[-4:] if len(key) > 10 else "(short key)"
            lines.append(f"  - {p['name']} [{p['api_key_env']}]: set ({masked})")
            for m in _models_for(p):
                dead_note = "  -- marked DEAD this run" if (p["name"], m) in _DEAD_COMBOS else ""
                lines.append(f"      · {m}{dead_note}")
        else:
            lines.append(f"  - {p['name']} [{p['api_key_env']}]: not set")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level LLM call, with automatic provider fallback
# ---------------------------------------------------------------------------

def _looks_like_payload_too_large(status_code, body_text):
    if status_code == 413:
        return True
    body_lower = (body_text or "").lower()
    return any(
        phrase in body_lower
        for phrase in (
            "payload too large", "request too large", "too many tokens",
            "context length", "context_length_exceeded", "maximum context length",
            "message too long", "exceeds the maximum",
        )
    )


class ModelUnavailableError(GroqCallError):
    """A specific MODEL was rejected -- decommissioned/deprecated, unknown
    slug, or not enabled for this key/tier (a 400/404 whose body says as
    much). Not a rate-limit issue, so no point retrying it: raised
    immediately, and it moves straight to the next model in the chain
    (same provider first, then the next provider) rather than burning
    retries against a model that's never going to answer."""
    pass


def _looks_like_model_unavailable(status_code, body_text):
    if status_code == 404:
        return True
    if status_code not in (400,):
        return False
    body_lower = (body_text or "").lower()
    return any(
        phrase in body_lower
        for phrase in (
            "model_decommissioned", "has been decommissioned", "has been deprecated",
            "does not exist", "model not found", "unknown model", "invalid model",
            "model_not_found", "not a valid model", "no longer supported",
        )
    )


def _call_llm_once(provider, model, messages, temperature, max_tokens, log, max_retries):
    """Does the actual HTTP call against ONE (provider, model) combo, with
    the same retry-on-429/timeout/network-error behavior the original
    single-provider call_groq() had. Raises ModelUnavailableError
    immediately (no retries) if the model itself is rejected as unknown/
    decommissioned; QuotaExhaustedError if every retry came back 429
    (reads as "tapped out"); ProviderAuthError on a 401/403 (bad/missing
    key); PayloadTooLargeError immediately on an oversized-payload
    response (never retried -- the caller re-chunks); or a plain
    GroqCallError for anything else that didn't recover within
    max_retries."""

    def _ts():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _log(msg):
        # Timestamp every line and hand it to whatever sink was passed in
        # (CLI's console log(), or the GUI's _forward_log -- both now print
        # immediately with flush=True, see below, so this is "live" in
        # both modes without double-printing).
        if log:
            log(f"[{_ts()}] {msg}")

    headers = {"Content-Type": "application/json"}
    if provider["api_key_env"] is not None:
        api_key = os.environ.get(provider["api_key_env"], "").strip()
        if not api_key:
            raise ProviderAuthError(f"{provider['name']}: no API key set ({provider['api_key_env']}).")
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    approx_chars = sum(len(m.get("content", "") or "") for m in messages if isinstance(m, dict))
    _log(f"-> POST {provider['endpoint']}  model={model}  "
         f"messages={len(messages)}  ~{approx_chars} chars  max_tokens={max_tokens}")

    limiter = _rate_limiter_for(provider["name"], model)
    last_exc = None
    saw_only_429 = True
    for attempt in range(1, max_retries + 1):
        limiter.wait_turn(_log)
        try:
            resp = requests.post(provider["endpoint"], headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
        except requests.exceptions.ConnectionError as exc:
            saw_only_429 = False
            last_exc = GroqCallError(f"{provider['name']}/{model}: connection failed: {exc}")
            _log(f"  ! {provider['name']}/{model} CONNECTION ERROR (attempt {attempt}/{max_retries}) -- "
                 f"is {provider['endpoint']} reachable from this machine right now (Tailscale up, "
                 f"host awake, Ollama running)? {exc}")
            time.sleep(1.5 * attempt)
            continue
        except requests.exceptions.Timeout:
            saw_only_429 = False
            last_exc = GroqCallError(f"{provider['name']}/{model}: took too long to respond.")
            _log(f"  ! {provider['name']}/{model} TIMEOUT after {GROQ_TIMEOUT_S}s (attempt {attempt}/{max_retries}) "
                 f"-- the model may still be loading into memory on first call, or the chunk is too big.")
            continue
        except requests.exceptions.RequestException as exc:
            saw_only_429 = False
            last_exc = GroqCallError(f"{provider['name']}/{model}: request failed: {exc}")
            _log(f"  ! {provider['name']}/{model} network error (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(1.5 * attempt)
            continue

        _log(f"<- HTTP {resp.status_code} from {provider['name']}/{model} "
             f"({len(resp.content or b'')} bytes)")

        if resp.status_code in (401, 403):
            _log(f"  ! auth error body: {resp.text[:400]}")
            raise ProviderAuthError(f"{provider['name']}: HTTP {resp.status_code} -- {resp.text[:200]}")

        if _looks_like_model_unavailable(resp.status_code, resp.text):
            _log(f"  ! {provider['name']}/{model} -- model unavailable "
                 f"(HTTP {resp.status_code}): {resp.text[:400]}  "
                 f"-- on Ollama this usually means the tag isn't pulled on THAT server yet; "
                 f"run `ollama pull {model}` on the host behind {OLLAMA_BASE_URL}.")
            raise ModelUnavailableError(f"{provider['name']}/{model}: HTTP {resp.status_code} -- {resp.text[:200]}")

        if resp.status_code == 429:
            limiter.note_rate_limited(_log)
            last_exc = GroqCallError(f"{provider['name']}/{model} 429: {resp.text[:300]}")
            _log(f"  ! {provider['name']}/{model} 429 (attempt {attempt}/{max_retries}): {resp.text[:300]}")
            time.sleep(1.0 * attempt)
            continue

        if _looks_like_payload_too_large(resp.status_code, resp.text):
            _log(f"  ! {provider['name']}/{model} says payload too large (HTTP {resp.status_code}) "
                 f"-- caller should clump smaller. body: {resp.text[:300]}")
            raise PayloadTooLargeError(resp.text[:300])

        if not resp.ok:
            saw_only_429 = False
            _log(f"  ! {provider['name']}/{model} error body: {resp.text[:600]}")
            raise GroqCallError(f"{provider['name']}/{model} error: HTTP {resp.status_code} {resp.text[:400]}")

        limiter.note_success()
        try:
            data = resp.json()
        except ValueError as exc:
            saw_only_429 = False
            last_exc = GroqCallError(f"{provider['name']}/{model}: response wasn't valid JSON: {exc}")
            _log(f"  ! {provider['name']}/{model} response wasn't valid JSON. Raw body (first 400 chars): "
                 f"{resp.text[:400]!r}")
            continue
        choices = data.get("choices") or []
        if not choices:
            _log(f"  ! {provider['name']}/{model} returned 200 but no 'choices' in the body: "
                 f"{json.dumps(data)[:400]}")
            return ""
        content = (choices[0].get("message", {}) or {}).get("content", "") or ""
        _log(f"  <- {len(content)} chars of content back from {provider['name']}/{model}")
        return content

    if saw_only_429:
        raise QuotaExhaustedError(
            f"{provider['name']}/{model}: {max_retries} straight 429(s) -- looks like the free "
            f"tier (or rate limit) for THIS MODEL is tapped out. {last_exc}"
        )
    raise last_exc or GroqCallError(f"{provider['name']}/{model}: call failed after retries.")


def warm_up_model(model=None, log=None):
    """Sends a tiny, throwaway prompt before the real run starts, purely so
    Ollama loads the model into memory NOW -- on a request small enough to
    finish fast -- instead of that one-time load cost getting silently
    added on top of the first REAL (often much larger) prompt's own
    processing time, which is what was tripping GROQ_TIMEOUT_S even after
    raising it. Failures here are logged but never raised; if the server's
    unreachable, the very next real call will surface that clearly anyway."""
    def _log(msg):
        if log:
            log(msg)
    if MANUAL_PASTE_MODE:
        _log("(manual paste mode is on -- skipping model warm-up, nothing to preload.)")
        return
    try:
        _log("(warming up model -- loading it into memory before the real run starts...)")
        call_groq(
            [{"role": "user", "content": "Reply with just: ok"}],
            temperature=0.0, max_tokens=8, model=model, log=log, max_retries=1,
        )
        _log("(model warm-up done)")
    except Exception as exc:
        _log(f"(model warm-up failed, continuing anyway -- the first real call will retry: {exc})")


def _show_manual_paste_dialog(parent, prompt_text, meta_line, result_holder):
    """Builds and shows the popup, and blocks (via mainloop or
    wait_window) until the user submits or aborts. MUST be called on the
    main thread. Writes its outcome into result_holder: either
    {"text": <pasted response>} or {"aborted": True}."""
    import tkinter as tk
    from tkinter import ttk

    BG, BG2, FG = "#14161b", "#1a1d24", "#eef0f4"
    ACCENT, GREEN, RED = "#33d1ff", "#33ff88", "#ff3b3b"

    own_root = None
    if parent is None:
        own_root = tk.Tk()
        win = own_root
    else:
        win = tk.Toplevel(parent)
        win.transient(parent)
        win.grab_set()

    win.title("Deepship — paste this into your LLM")
    win.configure(bg=BG)
    win.geometry("900x720")

    tk.Label(win, text=meta_line, bg=BG, fg=ACCENT, font=("Consolas", 10, "bold"),
             anchor="w", justify="left").pack(fill="x", padx=10, pady=(10, 4))
    tk.Label(win, text="1. Copy this prompt and paste it into your LLM of choice:",
             bg=BG, fg=FG, anchor="w").pack(fill="x", padx=10)

    prompt_frame = tk.Frame(win, bg=BG)
    prompt_frame.pack(fill="both", expand=True, padx=10, pady=(4, 4))
    prompt_scroll = tk.Scrollbar(prompt_frame)
    prompt_scroll.pack(side="right", fill="y")
    prompt_box = tk.Text(prompt_frame, wrap="word", height=16, bg=BG2, fg=FG,
                          insertbackground=FG, font=("Consolas", 10),
                          yscrollcommand=prompt_scroll.set)
    prompt_box.pack(side="left", fill="both", expand=True)
    prompt_scroll.configure(command=prompt_box.yview)
    prompt_box.insert("1.0", prompt_text)
    prompt_box.configure(state="disabled")

    def _copy():
        win.clipboard_clear()
        win.clipboard_append(prompt_text)
        copy_btn.configure(text="Copied to clipboard ✓")
        win.after(1200, lambda: copy_btn.configure(text="Copy prompt"))

    copy_btn = tk.Button(win, text="Copy prompt", command=_copy, bg=ACCENT, fg="#0a0b0d",
                          relief="flat", padx=12, pady=4, cursor="hand2")
    copy_btn.pack(padx=10, pady=(0, 10), anchor="e")

    ttk.Separator(win, orient="horizontal").pack(fill="x", padx=10, pady=4)

    tk.Label(win, text="2. Paste the LLM's response below, then click Submit:",
             bg=BG, fg=FG, anchor="w").pack(fill="x", padx=10, pady=(6, 4))

    resp_frame = tk.Frame(win, bg=BG)
    resp_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    resp_scroll = tk.Scrollbar(resp_frame)
    resp_scroll.pack(side="right", fill="y")
    response_box = tk.Text(resp_frame, wrap="word", height=12, bg=BG2, fg=FG,
                            insertbackground=FG, font=("Consolas", 10),
                            yscrollcommand=resp_scroll.set)
    response_box.pack(side="left", fill="both", expand=True)
    resp_scroll.configure(command=response_box.yview)
    response_box.focus_set()

    btn_row = tk.Frame(win, bg=BG)
    btn_row.pack(fill="x", padx=10, pady=(0, 10))

    def _close():
        if own_root is not None:
            own_root.destroy()
        else:
            win.grab_release()
            win.destroy()

    def _submit():
        text = response_box.get("1.0", "end-1c").strip()
        if not text:
            orig = response_box.cget("bg")
            response_box.configure(bg="#3a1a1a")
            win.after(300, lambda: response_box.configure(bg=orig))
            return
        result_holder["text"] = text
        _close()

    def _abort():
        result_holder["aborted"] = True
        _close()

    tk.Button(btn_row, text="Abort run", command=_abort, bg=RED, fg="white",
              relief="flat", padx=12, pady=6, cursor="hand2").pack(side="left")
    tk.Button(btn_row, text="Submit response", command=_submit, bg=GREEN, fg="#0a0b0d",
              relief="flat", padx=16, pady=6, cursor="hand2").pack(side="right")

    win.protocol("WM_DELETE_WINDOW", _abort)
    win.bind("<Control-Return>", lambda e: _submit())

    if own_root is not None:
        own_root.mainloop()
    else:
        parent.wait_window(win)


def manual_paste_call(messages, provider_label="manual", model_label="paste-mode", log=None):
    """The manual-paste-mode replacement for an actual LLM API call.
    Formats `messages` into one big prompt block, pops up a window with a
    Copy button so you can paste it into whatever LLM UI you want, and
    blocks until you paste the reply back and hit Submit (or Abort).
    Safe to call from the CLI's main thread OR the GUI's background
    crawler thread -- when a GUI is running (_GUI_ROOT is set) and this is
    called off the main thread, the popup is built on the main thread via
    root.after() and this function blocks on a threading.Event instead of
    touching Tkinter directly."""

    def _log(msg):
        if log:
            log(msg)

    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "user").upper()
        parts.append(f"===== {role} =====\n{m.get('content', '')}")
    prompt_text = "\n\n".join(parts)
    meta_line = (f"[{provider_label}/{model_label}]  {len(messages)} message(s), "
                 f"~{len(prompt_text)} chars -- copy below, paste your LLM's reply, Submit")

    _log(f"-> MANUAL PASTE MODE: opening popup, waiting for you to paste a response "
         f"({len(prompt_text)} chars prompt)")

    result = {}
    root = _GUI_ROOT
    if root is not None and threading.current_thread() is not threading.main_thread():
        done = threading.Event()

        def _run_on_main():
            try:
                _show_manual_paste_dialog(root, prompt_text, meta_line, result)
            finally:
                done.set()

        root.after(0, _run_on_main)
        done.wait()
    else:
        _show_manual_paste_dialog(root, prompt_text, meta_line, result)

    if result.get("aborted"):
        _log("  ! manual paste dialog aborted by user -- stopping run.")
        raise StopRequested("User clicked Abort in the manual paste dialog.")

    text = result.get("text", "")
    _log(f"  <- got {len(text)} chars pasted back in.")
    return text


def call_groq(messages, temperature=0.1, max_tokens=2000, model=None, log=None, max_retries=3):
    """PUBLIC entry point every pass in this pipeline calls (kept the name
    call_groq for backwards compatibility, but it no longer talks to Groq
    exclusively, and no longer talks to just one model per provider).
    Drives a flat chain of (provider, model) candidates built by
    _build_candidate_chain(): every model on Groq first (gpt-oss-120b,
    gpt-oss-20b, Qwen3.6-27b, ...), then every model on the next
    configured provider, and so on. Whichever combo last worked is tried
    first on the next call, so a live run doesn't re-probe dead combos
    over and over.

    If a call comes back as a rate-limit/quota tap-out
    (QuotaExhaustedError), an unknown/decommissioned model
    (ModelUnavailableError), a bad key (ProviderAuthError), or any other
    unrecovered error, that exact (provider, model) combo is marked dead
    for the rest of THIS run and the NEXT candidate in the chain is tried
    automatically with the identical prompt -- same swap logic whether
    that next candidate is a different model on the SAME provider (e.g.
    Groq's gpt-oss-120b -> Groq's gpt-oss-20b) or a completely different
    provider. Only raises once every configured (provider, model) combo
    has failed.

    PayloadTooLargeError is the one exception NOT treated as a fallback
    trigger -- it bubbles straight up so the caller's SPLIT_LADDER logic
    re-chunks and retries on the SAME combo, since a smaller chunk (not a
    different model/provider) is the actual fix.

    `model=` still works as a manual pin (e.g. --model from the CLI): it's
    inserted at the front of EVERY provider's model list, so it's tried
    first everywhere it might apply, but the rest of the chain is still
    there as a fallback if that exact model name isn't valid on a given
    provider."""

    def _log(msg):
        if log:
            log(msg)

    if MANUAL_PASTE_MODE:
        return manual_paste_call(messages, provider_label="manual-paste",
                                  model_label=(model or OLLAMA_MODEL), log=log)

    chain = _build_candidate_chain()
    if model:
        # A manual override applies provider-by-provider (a model name
        # valid on Groq usually isn't a valid slug on OpenAI), so just
        # prepend it once per provider rather than blindly to the whole
        # flat chain.
        chain = []
        for p in _configured_providers():
            models = [model] + [m for m in _models_for(p) if m != model]
            chain.extend((p, m) for m in models)

    if not chain:
        raise GroqCallError(f"No LLM provider API keys are set.\n{llm_provider_diagnostic()}")

    global _LAST_GOOD_COMBO
    with _PROVIDER_STATE_LOCK:
        sticky = _LAST_GOOD_COMBO

    sticky_key = (sticky[0]["name"], sticky[1]) if sticky else None
    if sticky and sticky_key not in _DEAD_COMBOS:
        rest = [c for c in chain if (c[0]["name"], c[1]) != sticky_key]
        order = [sticky] + rest
    else:
        order = chain

    # Try not-yet-dead combos first; if literally everything is dead
    # (every model on every provider has failed this run), give the whole
    # chain one more shot anyway rather than failing outright -- a daily
    # quota may have reset since it was marked dead.
    live = [c for c in order if (c[0]["name"], c[1]) not in _DEAD_COMBOS]
    order = live or order

    errors = []
    for i, (p, m) in enumerate(order):
        try:
            content = _call_llm_once(p, m, messages, temperature, max_tokens, log, max_retries)
            with _PROVIDER_STATE_LOCK:
                _LAST_GOOD_COMBO = (p, m)
            _DEAD_COMBOS.discard((p["name"], m))
            if i > 0:
                _log(f"  -> fell back to {p['name']}/{m} for this call.")
            else:
                _log(f"  -> using {p['name']}/{m}")
            return content
        except PayloadTooLargeError:
            raise
        except (QuotaExhaustedError, ModelUnavailableError, ProviderAuthError, GroqCallError) as exc:
            reason = "quota/rate-limit tapped out" if isinstance(exc, QuotaExhaustedError) else \
                      "model unavailable/decommissioned" if isinstance(exc, ModelUnavailableError) else \
                      "bad/missing API key" if isinstance(exc, ProviderAuthError) else "error"
            _log(f"  ! {p['name']}/{m} unavailable ({reason}): {exc}")
            _DEAD_COMBOS.add((p["name"], m))
            errors.append(f"{p['name']}/{m}: {exc}")
            if i < len(order) - 1:
                nxt_p, nxt_m = order[i + 1]
                same_provider = " (same provider, next model)" if nxt_p["name"] == p["name"] else ""
                _log(f"  -> auto-falling back to {nxt_p['name']}/{nxt_m}{same_provider}...")
            continue

    raise GroqCallError(
        "Every configured (provider, model) combo failed:\n" + "\n".join(errors) +
        "\n\n" + llm_provider_diagnostic()
    )




def _extract_json(text):
    """Strip ```json fences etc, same tolerant approach as the other scripts."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# HTTP fetch helper (requests-based, retried)
# ---------------------------------------------------------------------------

def fetch_url(url, log=None, max_retries=3, timeout=REQUEST_TIMEOUT_S):
    def _log(msg):
        if log:
            log(msg)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            _log(f"  ! fetch failed ({attempt}/{max_retries}) {url}: {exc}")
            time.sleep(1.0 * attempt)
    raise last_exc


def trim_to_domain(url: str) -> str:
    url = url.strip()
    p = urlparse(url)
    if not p.scheme:
        p = urlparse("https://" + url)
    if not p.netloc:
        raise ValueError(f"Could not parse a domain out of: {url!r}")
    return f"{p.scheme}://{p.netloc}"


# ---------------------------------------------------------------------------
# Step 1: crawl home page links
# ---------------------------------------------------------------------------

def crawl_homepage_links(homepage_url, log=None, on_event=None):
    """Fetches the home page and pulls out every <a href> link via
    BeautifulSoup, resolved to absolute URLs, deduped, paired with their
    visible anchor text (helps Groq's triage step). Fires a "links_found"
    event (homepage, links) for GUI consumers."""
    def _log(msg):
        if log:
            log(msg)

    html = fetch_url(homepage_url, log=log)
    soup = BeautifulSoup(html, "html.parser")

    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        full = urljoin(homepage_url, href)
        full = full.split("#")[0]
        if full in seen:
            continue
        seen.add(full)
        text = a.get_text(" ", strip=True)
        links.append({"url": full, "text": text[:120]})

    _log(f"Found {len(links)} unique link(s) on {homepage_url}")
    if on_event:
        on_event("links_found", homepage=homepage_url, links=links, html_len=len(html))
    return links


# ---------------------------------------------------------------------------
# Step 2: ask Groq which links look like financial aid / scholarships
# ---------------------------------------------------------------------------

LINK_TRIAGE_SYSTEM = (
    "You are helping a crawler find two kinds of pages on a college "
    "website, given only its home page's links. You will be given a JSON "
    "list of {url, text} pairs scraped from the home page. Pick every "
    "link that plausibly leads to EITHER of these: "
    "(1) kind \"finaid\" -- financial aid, scholarships, cost of "
    "attendance, tuition assistance, grants, or student financial "
    "services -- include a link even if it's just a hub/menu page that "
    "probably links onward to the real scholarships page; "
    "(2) kind \"faq\" -- a Frequently Asked Questions page, whether it's a "
    "financial-aid-specific FAQ or a general/admissions FAQ that likely "
    "touches on cost, aid, or scholarships. "
    "Respond ONLY with raw JSON (no prose, no markdown fences): a JSON "
    "array of objects, ranked best-guess first, each shaped exactly like "
    '{"url": "...", "reason": "short reason", "confidence": "high|medium|low", '
    '"kind": "finaid|faq"}. '
    "If truly nothing on the list looks relevant, respond with an empty "
    "JSON array: []"
)


def ask_groq_for_finaid_links(homepage_url, links, log=None, on_event=None, model=None):
    trimmed = links[:MAX_LINKS_SENT_TO_GROQ]
    user_content = (
        f"Home page: {homepage_url}\n\n"
        f"Links found on this page (JSON):\n{json.dumps(trimmed, ensure_ascii=False)}"
    )
    if on_event:
        on_event("link_triage_request", homepage=homepage_url, link_count=len(trimmed))
    content = call_groq(
        [
            {"role": "system", "content": LINK_TRIAGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=1500,
        model=model,
        log=log,
    )
    parsed = _extract_json(content)
    if parsed is None:
        if log:
            log("  ! Groq's link-triage reply wasn't valid JSON -- treating as no candidates.")
        if on_event:
            on_event("link_triage_response", homepage=homepage_url, raw=content, candidates=[])
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates") or parsed.get("links") or []
    if not isinstance(parsed, list):
        parsed = []

    valid_urls = {l["url"] for l in trimmed}
    out = []
    for item in parsed:
        if isinstance(item, str):
            item = {"url": item, "reason": "", "confidence": "medium", "kind": "finaid"}
        item.setdefault("kind", "finaid")
        if item.get("kind") not in ("finaid", "faq"):
            item["kind"] = "finaid"
        url = (item or {}).get("url")
        if url and url in valid_urls:
            out.append(item)

    if on_event:
        on_event("link_triage_response", homepage=homepage_url, raw=content, candidates=out)
    return out


# ---------------------------------------------------------------------------
# Step 3: fetch a candidate page and extract its visible text
# ---------------------------------------------------------------------------

def extract_page_text(html, log=None):
    """Same approach as alting_ua.py's extract_page_text(): strip
    script/style/noscript/template tags, then collapse whitespace so the
    text reads like what's actually on screen."""
    def _log(msg):
        if log:
            log(msg)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    raw_text = soup.get_text("\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(line for line in lines if line)
    _log(f"  BeautifulSoup extracted {len(text)} characters of visible text.")
    return text


# ---------------------------------------------------------------------------
# Step 4: adaptive chunked extraction (the 2 -> 4 -> 6 clumping ladder)
# ---------------------------------------------------------------------------

def chunk_text(text, n):
    """Split text into n roughly-equal-size chunks, breaking on paragraph
    boundaries (blank lines) where possible so a scholarship entry isn't
    sliced mid-sentence any more than necessary."""
    if n <= 1:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < n:
        size = max(1, len(text) // n)
        return [text[i:i + size] for i in range(0, len(text), size)] or [text]

    target_len = max(1, len(text) // n)
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        current.append(para)
        current_len += len(para)
        if current_len >= target_len and len(chunks) < n - 1:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Step 4b: Pass 4 -- pure-regex cross-check (no Groq call at all)
# ---------------------------------------------------------------------------
# This re-scans the *raw* extract_page_text() output -- the same
# BeautifulSoup visible-text passes 1/2/3 work from -- for whole paragraphs
# that mention financial aid in some form. It's a mechanical safety net:
# passes 1-3 all go through an LLM and can miss or mis-summarize things, so
# pass 4 gives a plain-text list of every paragraph that *looks* relevant by
# keyword alone, for a human to compare against what the LLM actually wrote
# up. It never calls Groq and never rewrites anything -- paragraphs are kept
# verbatim, in full.

PASS4_KEYWORDS = [
    r"financial aid", r"financial assistance", r"need-based", r"need based",
    r"need-blind", r"need blind", r"\baid\b", r"scholarships?", r"grants?",
    r"fellowships?", r"tuition assistance", r"tuition waiver", r"tuition-free",
    r"work[- ]study", r"stipends?", r"bursar(?:y|ies)", r"cost of attendance",
    r"\bFAFSA\b", r"CSS Profile", r"Pell Grant", r"full[- ]ride",
    r"full[- ]tuition", r"merit[- ]based", r"merit aid", r"loan forgiveness",
    r"student loans?", r"endowed scholarship", r"\bawards?\b", r"waivers?",
    r"subsid(?:y|ies)", r"discounted tuition", r"free tuition", r"free ride",
    r"low[- ]income", r"first[- ]generation", r"deadline", r"eligib(?:le|ility)",
    r"apply for aid", r"application fee waiver",
]

PASS4_REGEX = re.compile("|".join(PASS4_KEYWORDS), re.IGNORECASE)

_PASS4_WORD_RE = re.compile(r"[a-z0-9$%]+")


def _pass4_tokenset(s):
    return set(_PASS4_WORD_RE.findall((s or "").lower()))


def _pass4_normalize(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def regex_sweep_paragraphs(text, source_label=""):
    """PASS 4 core: splits `text` into paragraphs on blank-line boundaries
    (same split chunk_text() uses) and returns every WHOLE paragraph that
    contains one or more PASS4_KEYWORDS hits, each tagged with which
    keyword(s) matched and where it came from. Pure regex -- no Groq."""
    if not text or not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        hits = sorted(set(m.group(0).lower() for m in PASS4_REGEX.finditer(para)))
        if hits:
            out.append({"source": source_label, "paragraph": para, "matched_keywords": hits})
    return out


def run_pass4_regex_sweep(result, overlap_threshold=0.35, log=None, on_event=None):
    """PASS 4. No Groq calls. Re-scans the raw text already stashed on
    result["pages"][*]["raw_text"] and result["faq_raw_text"] (set during
    passes 1-3) for keyword-bearing paragraphs, dedupes them, and flags any
    paragraph whose distinctive words barely show up in pass 3's (or pass
    2's, if there was no FAQ gap-fill) combined structured output -- a
    cheap word-overlap heuristic for "does the LLM output already cover
    this, or did it possibly get missed / hallucinated over". Sets
    result["pass4_paragraphs"] (everything found) and
    result["pass4_flagged_new_info"] (the low-overlap subset)."""
    def _log(msg):
        if log:
            log(msg)

    domain = result.get("college", "")
    if on_event:
        on_event("pass4_start", domain=domain)

    all_paragraphs = []
    seen = set()

    for page in result.get("pages", []):
        for item in regex_sweep_paragraphs(page.get("raw_text", ""), source_label=page.get("url", "")):
            key = _pass4_normalize(item["paragraph"])
            if key in seen:
                continue
            seen.add(key)
            all_paragraphs.append(item)

    faq_raw = result.get("faq_raw_text", "")
    if faq_raw:
        faq_label = result.get("faq_url", "") or "(faq page)"
        for item in regex_sweep_paragraphs(faq_raw, source_label=faq_label):
            key = _pass4_normalize(item["paragraph"])
            if key in seen:
                continue
            seen.add(key)
            all_paragraphs.append(item)

    # "The entire output" to compare against -- pass 3's FAQ-enriched
    # entries if there are any, otherwise fall back to pass 2's combined
    # structured entries.
    pass3_entries = result.get("scholarships_faq_enriched") or \
        [e for p in result.get("pages", []) for e in p.get("scholarships", [])]
    pass3_blob = " ".join(
        f"{e.get('type', '')} {e.get('summary', '')}"
        for e in pass3_entries if isinstance(e, dict)
    )
    pass3_tokens = _pass4_tokenset(pass3_blob)

    flagged = []
    for item in all_paragraphs:
        distinctive = {t for t in _pass4_tokenset(item["paragraph"]) if len(t) > 3}
        if not distinctive:
            continue
        overlap = len(distinctive & pass3_tokens) / len(distinctive)
        item["overlap_with_pass3"] = round(overlap, 2)
        if overlap < overlap_threshold:
            flagged.append(item)

    result["pass4_paragraphs"] = all_paragraphs
    result["pass4_flagged_new_info"] = flagged

    _log(f"  -> pass 4 (regex, no Groq): {len(all_paragraphs)} matching paragraph(s) found; "
         f"{len(flagged)} look like they may have info not reflected in pass 3's output.")

    if on_event:
        on_event("pass4_done", domain=domain, paragraphs=all_paragraphs, flagged=flagged)

    return all_paragraphs, flagged


# ---------------------------------------------------------------------------
# PASS 1 -- loose sweep: pull out any bits of text that LOOK LIKE they might
# be a scholarship / need-based aid / full-ride mention. No structuring yet,
# no deadline-parsing, no filtering for "is this really a distinct award" --
# just cast a wide net and grab the raw text so nothing gets missed. Pass 2
# (below) is what actually tries to turn this into a clean record.
# ---------------------------------------------------------------------------

PASS1_SNIPPET_SYSTEM = (
    "You are doing a FIRST, LOOSE pass over a chunk of a college financial "
    "aid / scholarships web page. Do NOT try to build a structured "
    "scholarship record yet, and do NOT worry about deadlines, exact "
    "amounts, or whether something is 'really' a distinct award -- that "
    "comes in a later pass. Right now just find every piece of text that "
    "plausibly mentions a scholarship, grant, need-based aid, or "
    "full-ride/full-tuition award, and pull it out close to verbatim from "
    "the page. Cast a wide net: include anything that even hints at a "
    "named award, a dollar figure, an eligibility rule, or a deadline, "
    "even if it's fragmentary, vague, or you're not fully sure it's real "
    "scholarship content. Over-including is fine; a second pass will sort "
    "the good hits from the noise. "
    "Respond ONLY with raw JSON (no prose, no markdown fences): a JSON "
    "array of objects shaped exactly like "
    '{"snippet": "the pulled text, verbatim or near-verbatim from the '
    'page", "hint": "one short phrase guessing what this might be, e.g. '
    '\'possible need-based grant\' or \'possible full-ride program\' or '
    "'mentions a deadline'\"}. "
    "If this chunk has nothing that even loosely resembles scholarship/aid "
    "content, respond with an empty JSON array: []"
)


def _build_pass1_prompt(chunk_text_, source_url, chunk_index, total_chunks, prev_snippets_json):
    header = f"Source URL: {source_url}\nChunk {chunk_index + 1} of {total_chunks}.\n\n"
    if chunk_index == 0:
        context_note = ""
    else:
        context_note = (
            "Snippets already pulled from the PREVIOUS chunk of this same "
            "page (for continuity only -- this earlier list may be "
            "incomplete because it was cut off at the chunk boundary):\n"
            f"{prev_snippets_json}\n\n"
            "This new chunk continues directly after that one. Don't "
            "re-pull a snippet already captured above; if the previous "
            "snippet was clearly cut off and this chunk continues it, pull "
            "the continuation as its own snippet rather than trying to "
            "merge them yourself (merging happens in a later pass).\n\n"
        )
    return header + context_note + f"Page text chunk:\n{chunk_text_}"


def collect_scholarship_snippets_from_page(source_url, page_text, log=None, model=None,
                                            on_event=None, stop_flag=None):
    """PASS 1. Same adaptive 2 -> 4 -> 6 clumping ladder as before (to dodge
    413s on big pages), but instead of asking Groq to structure anything, it
    just asks for loosely-collected raw snippets. Returns a deduped list of
    {snippet, hint} dicts. Fires pass1_chunk_split / pass1_chunk_request /
    pass1_chunk_response / payload_too_large events for GUI consumers."""
    def _log(msg):
        if log:
            log(msg)

    for split_n in SPLIT_LADDER:
        _check_stop(stop_flag)
        chunks = chunk_text(page_text, split_n)
        _log(f"  [pass 1] Trying {source_url} split into {len(chunks)} chunk(s)...")
        if on_event:
            on_event("pass1_chunk_split", url=source_url, split_n=split_n, chunk_count=len(chunks))
        try:
            all_snippets = []
            prev_snippets_json = "[]"
            for idx, chunk in enumerate(chunks):
                _check_stop(stop_flag)
                if on_event:
                    on_event("pass1_chunk_request", url=source_url, chunk_index=idx, total_chunks=len(chunks))
                prompt = _build_pass1_prompt(chunk, source_url, idx, len(chunks), prev_snippets_json)
                content = call_groq(
                    [
                        {"role": "system", "content": PASS1_SNIPPET_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=2500,
                    model=model,
                    log=log,
                )
                parsed = _extract_json(content)
                if parsed is None:
                    _log(f"  ! [pass 1] chunk {idx + 1}/{len(chunks)} wasn't valid JSON, skipping it.")
                    parsed = []
                if isinstance(parsed, dict):
                    parsed = parsed.get("snippets") or parsed.get("entries") or []
                if not isinstance(parsed, list):
                    parsed = []
                all_snippets.extend(parsed)
                if on_event:
                    on_event("pass1_chunk_response", url=source_url, chunk_index=idx, total_chunks=len(chunks),
                              raw=content, snippets=parsed)
                prev_snippets_json = json.dumps(parsed[-3:], ensure_ascii=False)

            return _dedupe_snippets(all_snippets)

        except PayloadTooLargeError:
            _log(f"  [pass 1] Payload too large at split={split_n} -- escalating to a finer split.")
            if on_event:
                on_event("payload_too_large", url=source_url, split_n=split_n, pass_num=1)
            continue

    _check_stop(stop_flag)
    _log(f"  ! [pass 1] Still too large after {SPLIT_LADDER[-1]}-way split -- falling back to a truncated single chunk.")
    truncated = page_text[: max(1, len(page_text) // (SPLIT_LADDER[-1] * 2))]
    try:
        content = call_groq(
            [
                {"role": "system", "content": PASS1_SNIPPET_SYSTEM},
                {"role": "user", "content": _build_pass1_prompt(truncated, source_url, 0, 1, "[]")},
            ],
            temperature=0,
            max_tokens=2500,
            model=model,
            log=log,
        )
        parsed = _extract_json(content) or []
        if isinstance(parsed, dict):
            parsed = parsed.get("snippets") or parsed.get("entries") or []
        parsed = parsed if isinstance(parsed, list) else []
        if on_event:
            on_event("pass1_chunk_response", url=source_url, chunk_index=0, total_chunks=1,
                      raw=content, snippets=parsed)
        return _dedupe_snippets(parsed)
    except PayloadTooLargeError:
        _log(f"  ! [pass 1] Gave up on {source_url} -- even the truncated fallback was too large.")
        if on_event:
            on_event("payload_too_large", url=source_url, split_n=SPLIT_LADDER[-1], gave_up=True, pass_num=1)
        return []


def _dedupe_snippets(snippets):
    seen = set()
    out = []
    for s in snippets:
        if not isinstance(s, dict):
            continue
        text = (s.get("snippet") or "").strip()
        key = text.lower()[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# PASS 2 -- structuring: take the loose snippets pass 1 found for a page and
# ask Groq to turn them into clean, tighter-filtered scholarship records
# (name, eligibility, amount, deadline, etc). This is the "rerun with
# tighter filters" step -- it only ever looks at what pass 1 already pulled,
# it doesn't go back to the raw page text.
# ---------------------------------------------------------------------------

PASS2_STRUCTURE_SYSTEM = (
    "You are doing a SECOND pass over loosely-collected snippets (pass 1) "
    "that were already flagged as possibly describing a scholarship, "
    "grant, or financial-aid program. Do NOT force these into a rigid "
    "table with strict fields like exact amount/deadline columns -- this "
    "isn't a database record, it's a rough, readable summary someone will "
    "scan on a little scrollable card. Group snippets that clearly "
    "describe the SAME program together; drop snippets that, on closer "
    "look, aren't really describing a real program (nav text, generic "
    "'contact financial aid' boilerplate, etc). "
    "For EVERY distinct program you can identify, produce one object "
    "shaped exactly like: "
    '{"type": "...", "summary": "..."}. '
    "For \"type\", pick whichever loose category fits best (you can "
    "combine a couple with a comma): need-based, need-blind, merit, "
    "full-ride, full-tuition, athletic, departmental, or other. "
    "For \"summary\", write 1-3 plain-language sentences a person could "
    "skim quickly -- what the program is called (if named), roughly who "
    "it's for, and casually work in anything about amount, deadline, or "
    "how to apply if the snippets mention it (e.g. \"The Founders "
    "Scholarship looks like a need-based award covering full tuition for "
    "first-gen students -- snippets mention an early-January deadline.\"). "
    "Don't invent specifics that aren't in the snippets; if something's "
    "unclear just say so in plain words instead of guessing a number. "
    "Respond ONLY with a raw JSON array of these objects (no prose, no "
    "markdown fences). If none of the snippets actually describe a real "
    "program once you look closely, respond with an empty JSON array: []"
)


def structure_snippets_for_page(source_url, snippets, log=None, model=None, on_event=None, stop_flag=None):
    """PASS 2. Sends the snippets pass 1 collected for one page back to
    Groq to be deduped/merged/structured into real records. Chunks the
    snippet list itself (not the raw page) if it's large, using the same
    413-triggered escalation as pass 1. Returns a deduped list of
    structured scholarship dicts. Fires pass2_request / pass2_response /
    payload_too_large events."""
    def _log(msg):
        if log:
            log(msg)

    if not snippets:
        return []

    def _call_pass2(batch, batch_index, total_batches):
        _check_stop(stop_flag)
        if on_event:
            on_event("pass2_request", url=source_url, batch_index=batch_index,
                      total_batches=total_batches, snippet_count=len(batch))
        user_content = (
            f"Source URL: {source_url}\n"
            f"Snippet batch {batch_index + 1} of {total_batches} "
            f"({len(batch)} snippet(s) from pass 1):\n\n"
            f"{json.dumps(batch, ensure_ascii=False)}"
        )
        content = call_groq(
            [
                {"role": "system", "content": PASS2_STRUCTURE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=2500,
            model=model,
            log=log,
        )
        parsed = _extract_json(content)
        if parsed is None:
            _log(f"  ! [pass 2] batch {batch_index + 1}/{total_batches} wasn't valid JSON, skipping it.")
            parsed = []
        if isinstance(parsed, dict):
            parsed = parsed.get("scholarships") or parsed.get("entries") or []
        if not isinstance(parsed, list):
            parsed = []
        if on_event:
            on_event("pass2_response", url=source_url, batch_index=batch_index,
                      total_batches=total_batches, raw=content, entries=parsed)
        return parsed

    for split_n in [1] + SPLIT_LADDER:
        _check_stop(stop_flag)
        batch_size = max(1, -(-len(snippets) // split_n))  # ceil div
        batches = [snippets[i:i + batch_size] for i in range(0, len(snippets), batch_size)] or [snippets]
        try:
            all_entries = []
            for idx, batch in enumerate(batches):
                all_entries.extend(_call_pass2(batch, idx, len(batches)))
            return _dedupe_scholarship_entries(all_entries)
        except PayloadTooLargeError:
            _log(f"  [pass 2] Payload too large batching snippets {split_n}-way -- escalating.")
            if on_event:
                on_event("payload_too_large", url=source_url, split_n=split_n, pass_num=2)
            continue

    _log(f"  ! [pass 2] Gave up on {source_url} -- even {SPLIT_LADDER[-1]}-way snippet batching was too large.")
    return []


def _dedupe_scholarship_entries(entries):
    seen = set()
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        key = (e.get("summary") or "").strip().lower()[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def extract_scholarships_from_page(source_url, page_text, log=None, model=None, on_event=None, stop_flag=None):
    """Orchestrates PASS 1 (loose snippet sweep) then PASS 2 (structuring).
    Returns (snippets, structured_entries) -- both are shown to the user;
    pass 1's snippets are the quick first look, pass 2 is the tighter,
    structured re-run over just what pass 1 flagged."""
    snippets = collect_scholarship_snippets_from_page(
        source_url, page_text, log=log, model=model, on_event=on_event, stop_flag=stop_flag)
    _check_stop(stop_flag)
    structured = structure_snippets_for_page(
        source_url, snippets, log=log, model=model, on_event=on_event, stop_flag=stop_flag)
    return snippets, structured


# ---------------------------------------------------------------------------
# PASS 3 -- FAQ gap-fill: after pass 2 has rough type+summary cards for a
# college, use that college's FAQ page (if the link-triage step found one)
# to fill in anything a card is missing -- a deadline, an amount, an
# eligibility detail -- and to pick up any program that's only mentioned on
# the FAQ page and nowhere else. This never invents; it only adds what the
# FAQ text actually says, and only touches the {type, summary} shape pass 2
# already produced.
# ---------------------------------------------------------------------------

GAP_FILL_SYSTEM = (
    "You are given a list of rough scholarship/financial-aid program "
    "summaries that were already written up (each has a loose \"type\" "
    "and a plain-language \"summary\"), plus a chunk of this same "
    "college's FAQ page text. Your job is to fill in GAPS in the existing "
    "summaries using anything relevant on this FAQ chunk -- a deadline, an "
    "amount, an eligibility detail, a renewal rule, or how to apply, that "
    "wasn't already mentioned. Do not invent anything not actually stated "
    "in the FAQ text. If the FAQ chunk doesn't add anything useful for a "
    "given program, leave that program's summary as it was (light wording "
    "cleanup is fine, but don't shorten it or drop real content). If this "
    "FAQ chunk clearly describes a genuinely NEW program that wasn't in "
    "the list at all, add it as a new {type, summary} entry. "
    "Respond ONLY with a raw JSON array covering ALL programs -- the "
    "existing ones (updated where the FAQ helped) plus any new ones found "
    "-- each shaped exactly like {\"type\": \"...\", \"summary\": \"...\"}. "
    "No prose, no markdown fences."
)


def _build_gap_fill_prompt(faq_chunk, faq_url, chunk_index, total_chunks, current_entries_json):
    header = f"FAQ page: {faq_url}\nFAQ chunk {chunk_index + 1} of {total_chunks}.\n\n"
    body = (
        "Program summaries so far (from pass 2, to be enriched -- carry "
        "every one of these forward in your reply, updated or unchanged):\n"
        f"{current_entries_json}\n\n"
        f"FAQ page text chunk:\n{faq_chunk}"
    )
    return header + body


def gap_fill_with_faq(college_domain, entries, faq_url, faq_text, log=None, model=None,
                       on_event=None, stop_flag=None):
    """PASS 3. Walks the FAQ page text in the same 2 -> 4 -> 6 chunk ladder
    as passes 1/2, each time asking Groq to enrich the *running* list of
    {type, summary} entries with anything new the FAQ chunk adds. Returns
    the enriched, deduped entry list (or the original entries unchanged if
    there's no FAQ text or every attempt fails). Fires gap_fill_request /
    gap_fill_response / payload_too_large events."""
    def _log(msg):
        if log:
            log(msg)

    if not faq_text or not faq_text.strip() or not entries:
        return entries

    for split_n in SPLIT_LADDER:
        _check_stop(stop_flag)
        chunks = chunk_text(faq_text, split_n)
        _log(f"  [pass 3 / FAQ gap-fill] Trying {faq_url} split into {len(chunks)} chunk(s)...")
        try:
            working = list(entries)
            for idx, chunk in enumerate(chunks):
                _check_stop(stop_flag)
                if on_event:
                    on_event("gap_fill_request", domain=college_domain, url=faq_url,
                              chunk_index=idx, total_chunks=len(chunks))
                prompt = _build_gap_fill_prompt(chunk, faq_url, idx, len(chunks),
                                                 json.dumps(working, ensure_ascii=False))
                content = call_groq(
                    [
                        {"role": "system", "content": GAP_FILL_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=3000,
                    model=model,
                    log=log,
                )
                parsed = _extract_json(content)
                if isinstance(parsed, dict):
                    parsed = parsed.get("scholarships") or parsed.get("entries") or []
                if isinstance(parsed, list) and parsed:
                    working = parsed
                else:
                    _log(f"  ! [pass 3] FAQ chunk {idx + 1}/{len(chunks)} wasn't usable JSON -- keeping prior entries.")
                if on_event:
                    on_event("gap_fill_response", domain=college_domain, url=faq_url,
                              chunk_index=idx, total_chunks=len(chunks), raw=content, entries=working)
            return _dedupe_scholarship_entries(working)
        except PayloadTooLargeError:
            _log(f"  [pass 3] Payload too large at split={split_n} -- escalating to a finer split.")
            if on_event:
                on_event("payload_too_large", url=faq_url, split_n=split_n, pass_num=3)
            continue

    _log(f"  ! [pass 3] Gave up on {faq_url} -- even {SPLIT_LADDER[-1]}-way split was too large; "
         f"leaving pass-2 entries as-is.")
    return entries


# ---------------------------------------------------------------------------
# Orchestration: one college end-to-end
# ---------------------------------------------------------------------------

def process_college(domain, max_candidates=DEFAULT_MAX_CANDIDATES, model=None, log=None,
                     on_event=None, stop_flag=None, index=None, total=None):
    def _log(msg):
        if log:
            log(msg)
        elif on_event is None:
            print(msg)

    homepage_url = trim_to_domain(domain)
    result = {"college": domain, "homepage": homepage_url, "candidate_links": [], "pages": [],
              "faq_raw_text": "", "pass4_paragraphs": [], "pass4_flagged_new_info": []}

    if on_event:
        on_event("college_start", domain=domain, homepage=homepage_url, index=index, total=total)
    else:
        _log(f"\n=== {domain} ===")

    try:
        _check_stop(stop_flag)
        links = crawl_homepage_links(homepage_url, log=_log, on_event=on_event)
    except StopRequested:
        raise
    except Exception as exc:
        _log(f"  ! failed to crawl home page: {exc}")
        result["error"] = f"homepage crawl failed: {exc}"
        if on_event:
            on_event("college_done", domain=domain, candidate_count=0, page_count=0,
                      scholarship_count=0, error=str(exc))
        return result

    if not links:
        result["error"] = "no links found on home page"
        if on_event:
            on_event("college_done", domain=domain, candidate_count=0, page_count=0,
                      scholarship_count=0, error=result["error"])
        return result

    time.sleep(POLITE_DELAY_S)
    _check_stop(stop_flag)

    try:
        candidates = ask_groq_for_finaid_links(homepage_url, links, log=_log, on_event=on_event, model=model)
    except StopRequested:
        raise
    except (GroqCallError, PayloadTooLargeError) as exc:
        _log(f"  ! Groq link-triage failed: {exc}")
        result["error"] = f"link triage failed: {exc}"
        if on_event:
            on_event("college_done", domain=domain, candidate_count=0, page_count=0,
                      scholarship_count=0, error=str(exc))
        return result

    result["candidate_links"] = candidates
    finaid_candidates = [c for c in candidates if c.get("kind", "finaid") == "finaid"]
    faq_candidates = [c for c in candidates if c.get("kind") == "faq"]

    if on_event:
        on_event("candidates_selected", domain=domain, candidates=finaid_candidates[:max_candidates])
        if faq_candidates:
            on_event("faq_candidate_selected", domain=domain, candidate=faq_candidates[0])

    if not finaid_candidates:
        _log("  Groq found no plausible financial aid / scholarships link on the home page.")
        if on_event:
            on_event("college_done", domain=domain, candidate_count=0, page_count=0, scholarship_count=0)
        return result

    _log(f"  Groq flagged {len(finaid_candidates)} finaid candidate link(s), deep-diving top {max_candidates}:")
    for c in finaid_candidates[:max_candidates]:
        _log(f"    - [{c.get('confidence', '?')}] {c.get('url')} :: {c.get('reason', '')}")
    if faq_candidates:
        _log(f"  Groq also flagged an FAQ page to use for gap-filling: {faq_candidates[0].get('url')}")

    for candidate in finaid_candidates[:max_candidates]:
        _check_stop(stop_flag)
        url = candidate.get("url")
        if not url:
            continue
        time.sleep(POLITE_DELAY_S)
        page_record = {"url": url, "confidence": candidate.get("confidence"),
                        "reason": candidate.get("reason"), "pass1_snippets": [], "scholarships": [],
                        "raw_text": ""}
        if on_event:
            on_event("page_start", domain=domain, url=url, confidence=candidate.get("confidence"),
                      reason=candidate.get("reason"))
        try:
            html = fetch_url(url, log=_log)
            if on_event:
                on_event("page_fetched", domain=domain, url=url, html_len=len(html))
            text = extract_page_text(html, log=_log)
            page_record["raw_text"] = text
            if on_event:
                on_event("page_text_extracted", domain=domain, url=url, text=text)
            if not text.strip():
                _log(f"  ! {url} had no visible text after extraction, skipping.")
                result["pages"].append(page_record)
                if on_event:
                    on_event("page_done", domain=domain, url=url, snippets=[], entries=[])
                continue
            snippets, entries = extract_scholarships_from_page(url, text, log=_log, model=model,
                                                                 on_event=on_event, stop_flag=stop_flag)
            page_record["pass1_snippets"] = snippets
            page_record["scholarships"] = entries
            _log(f"  -> pass 1: {len(snippets)} possible snippet(s)  |  "
                 f"pass 2: {len(entries)} structured entr{'y' if len(entries) == 1 else 'ies'} on {url}")
            if on_event:
                on_event("page_done", domain=domain, url=url, snippets=snippets, entries=entries)
        except StopRequested:
            result["pages"].append(page_record)
            raise
        except Exception as exc:
            _log(f"  ! failed processing {url}: {exc}")
            page_record["error"] = str(exc)
            if on_event:
                on_event("page_done", domain=domain, url=url, snippets=[], entries=[], error=str(exc))
        result["pages"].append(page_record)

    # PASS 3 -- FAQ gap-fill: if Groq flagged an FAQ page, fetch it and use
    # it to fill in gaps in the pass-2 summaries gathered above.
    combined_entries = [e for p in result["pages"] for e in p.get("scholarships", [])]
    result["faq_url"] = None
    result["scholarships_faq_enriched"] = []
    if faq_candidates and combined_entries:
        faq_url = faq_candidates[0].get("url")
        result["faq_url"] = faq_url
        _check_stop(stop_flag)
        time.sleep(POLITE_DELAY_S)
        try:
            if on_event:
                on_event("faq_fetch_start", domain=domain, url=faq_url)
            faq_html = fetch_url(faq_url, log=_log)
            faq_text = extract_page_text(faq_html, log=_log)
            result["faq_raw_text"] = faq_text
            if on_event:
                on_event("faq_fetched", domain=domain, url=faq_url, text_len=len(faq_text))
            if faq_text.strip():
                enriched = gap_fill_with_faq(domain, combined_entries, faq_url, faq_text, log=_log,
                                              model=model, on_event=on_event, stop_flag=stop_flag)
                result["scholarships_faq_enriched"] = enriched
                _log(f"  -> pass 3: FAQ gap-fill produced {len(enriched)} entr"
                     f"{'y' if len(enriched) == 1 else 'ies'} (from {len(combined_entries)} going in)")
                if on_event:
                    on_event("gap_fill_done", domain=domain, url=faq_url, entries=enriched)
            else:
                _log(f"  ! {faq_url} had no visible text after extraction, skipping FAQ gap-fill.")
        except StopRequested:
            raise
        except Exception as exc:
            _log(f"  ! FAQ gap-fill failed on {faq_url}: {exc}")
    elif faq_candidates and not combined_entries:
        _log("  (found an FAQ page, but no pass-2 entries to gap-fill yet -- skipping.)")

    # PASS 4 -- pure-regex cross-check of the raw BeautifulSoup text against
    # pass 3's (or pass 2's) structured output. No Groq call.
    try:
        run_pass4_regex_sweep(result, log=_log, on_event=on_event)
    except Exception as exc:
        _log(f"  ! pass 4 regex sweep failed: {exc}")

    scholarship_count = sum(len(p.get("scholarships", [])) for p in result["pages"])
    if on_event:
        on_event("college_done", domain=domain, candidate_count=len(candidates),
                  page_count=len(result["pages"]), scholarship_count=scholarship_count)
    return result


# ---------------------------------------------------------------------------
# GUI-side crawler wrapper (background thread + stop flag)
# ---------------------------------------------------------------------------

class DeepshipCrawler:
    """Runs process_college() for a batch of domains on a background
    thread, forwarding every pipeline event to on_event (which the GUI
    marshals onto its Tk event queue). stop() sets a threading.Event that
    every pipeline stage checks between steps, so Stop takes effect at the
    next safe point (end of current Groq call) rather than mid-request."""

    def __init__(self, on_event, max_candidates=DEFAULT_MAX_CANDIDATES, model=None):
        self.on_event = on_event
        self.max_candidates = max_candidates
        self.model = model
        self._stop = threading.Event()
        self._thread = None

    def _forward_log(self, msg):
        # Every provider/model/HTTP-error message call_groq() and
        # _call_llm_once() produce -- previously passed log=None here and
        # silently dropped in the GUI -- now reaches the live log panel.
        # Also echoed straight to the console (flushed immediately) so if
        # you're running the GUI from a terminal you see every request/
        # response line live there too, not just in the "Groq's Response"
        # pane.
        print(msg, flush=True)
        self.on_event("log_line", text=msg)

    def start_batch(self, domains):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(list(domains),), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self, domains):
        total = len(domains)
        self.on_event("run_start", total=total)
        warm_up_model(model=self.model, log=self._forward_log)
        total_scholarships = 0
        stopped_early = False
        for i, domain in enumerate(domains):
            if self._stop.is_set():
                stopped_early = True
                break
            try:
                result = process_college(
                    domain, max_candidates=self.max_candidates, model=self.model,
                    log=self._forward_log, on_event=self.on_event, stop_flag=self._stop,
                    index=i, total=total,
                )
                total_scholarships += sum(len(p.get("scholarships", [])) for p in result.get("pages", []))
            except StopRequested:
                stopped_early = True
                break
            except Exception as exc:
                self.on_event("college_done", domain=domain, candidate_count=0, page_count=0,
                              scholarship_count=0, error=str(exc))
        self.on_event("run_done", total_scholarships=total_scholarships, stopped_early=stopped_early)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    BG_DARK = "#0a0b0d"
    BG_PANEL = "#14161b"
    BG_PANEL2 = "#1a1d24"
    FG_DIM = "#7d8390"
    FG_BRIGHT = "#eef0f4"
    ACCENT_CYAN = "#33d1ff"
    ACCENT_PURPLE = "#b26bff"
    ACCENT_AMBER = "#ffb020"
    GREEN = "#33ff88"
    RED = "#ff3b3b"
    BORDER = "#2a2e37"
    BLUE = "#5b8cff"
    PINK = "#ff6bd6"
    GREY = "#9aa5b8"

    # The seven stages of the pipeline (per docstring at top of file), in
    # the order they fire for a given page/college -- drives the full-width
    # stage tracker bar at the top of the window. "fetch"/"pass1"/"pass2"
    # repeat once per candidate page; "pass3"/"pass4" run once per college
    # after all its pages are done.
    STAGE_ORDER = ["crawl", "triage", "fetch", "pass1", "pass2", "pass3", "pass4"]
    # Boxes actually drawn in the pipeline bar -- crawl/triage/fetch still run
    # and still fire all their events exactly as before (see STAGE_ORDER above
    # and every _advance_stage("crawl"/"triage"/"fetch") call below); they just
    # have no visible box to light up, so those calls become harmless no-ops.
    VISIBLE_STAGE_ORDER = ["pass1", "pass2", "pass3", "pass4"]
    STAGE_META = {
        "crawl":  {"label": "Crawl",             "color": ACCENT_CYAN,  "idle": "waiting…"},
        "triage": {"label": "Triage",            "color": ACCENT_PURPLE,"idle": "waiting…"},
        "fetch":  {"label": "Fetch Page",        "color": ACCENT_AMBER, "idle": "waiting…"},
        "pass1":  {"label": "Pass 1 · Sweep",    "color": GREEN,        "idle": "waiting…"},
        "pass2":  {"label": "Pass 2 · Structure","color": BLUE,         "idle": "waiting…"},
        "pass3":  {"label": "Pass 3 · FAQ",      "color": PINK,         "idle": "waiting…"},
        "pass4":  {"label": "Pass 4 · Regex",    "color": GREY,         "idle": "waiting…"},
    }

    MAX_TEXT_PREVIEW_CHARS = 20000
    MAX_LOG_LINES = 2000
    STAGE_MAX_LINES = 400  # per stage-box text widget; trimmed from the top

    def _paste_keys_popup():
        """Startup popup, shown before the main window: paste any number
        of API keys, one per line (or KEY=VALUE .env-style lines), in
        any order/provider mix. Each is auto-detected by its prefix and
        assigned straight into os.environ for this process -- sidesteps
        a broken .env entirely, since nothing is written to disk. Ends
        by showing which provider(s) were detected before moving on."""
        win = tk.Tk()
        win.title("Deepship — Paste API Keys")
        win.configure(bg=BG_DARK)
        win.geometry("580x480")
        win.resizable(False, False)
        try:
            win.eval("tk::PlaceWindow . center")
        except tk.TclError:
            pass

        tk.Label(win, text="Paste your API key(s) below, one per line.",
                 bg=BG_DARK, fg=FG_BRIGHT, font=("TkDefaultFont", 12, "bold")).pack(
            anchor="w", padx=16, pady=(16, 4))
        tk.Label(win,
                 text="Any provider, any order, any mix — Groq, Cerebras, Together,\n"
                      "Fireworks, OpenRouter, DeepSeek, OpenAI. Each key is\n"
                      "auto-detected by its prefix and assigned to the right\n"
                      "provider automatically. Plain KEY=value (.env-style) lines\n"
                      "work too. Nothing is written to disk.",
                 bg=BG_DARK, fg=FG_DIM, justify="left", font=("TkDefaultFont", 9)).pack(
            anchor="w", padx=16, pady=(0, 10))

        text = tk.Text(win, height=13, width=64, bg=BG_PANEL2, fg=FG_BRIGHT,
                        insertbackground=FG_BRIGHT, bd=0, highlightthickness=1,
                        highlightbackground=BORDER, font=("Consolas", 10), wrap="none")
        text.pack(padx=16, pady=(0, 10), fill="both", expand=True)
        text.focus_set()

        result = {"assigned": [], "unrecognized": []}

        def _do_continue(event=None):
            raw = text.get("1.0", "end")
            assigned, unrecognized = _assign_pasted_keys(raw)
            result["assigned"] = assigned
            result["unrecognized"] = unrecognized
            _refresh_groq_env()
            win.destroy()

        def _do_skip():
            win.destroy()

        btns = tk.Frame(win, bg=BG_DARK)
        btns.pack(fill="x", padx=16, pady=(0, 16))
        tk.Button(btns, text="Continue →", command=_do_continue, bg=BG_PANEL2, fg=GREEN,
                  activebackground="#242832", activeforeground=GREEN, bd=0, padx=14, pady=6,
                  highlightthickness=1, highlightbackground=BORDER).pack(side="left")
        tk.Button(btns, text="Skip (use existing / .env)", command=_do_skip, bg=BG_PANEL2, fg=FG_DIM,
                  activebackground="#242832", activeforeground=FG_BRIGHT, bd=0, padx=14, pady=6,
                  highlightthickness=1, highlightbackground=BORDER).pack(side="left", padx=8)

        win.bind("<Control-Return>", _do_continue)
        win.protocol("WM_DELETE_WINDOW", _do_skip)
        win.mainloop()

        assigned, unrecognized = result["assigned"], result["unrecognized"]
        if assigned:
            seen = set()
            lines = []
            for name, env_var in assigned:
                if name in seen:
                    continue
                seen.add(name)
                lines.append(f"  • {name}   [{env_var}]")
            summary = "Detected and assigned:\n\n" + "\n".join(lines)
            if unrecognized:
                summary += (f"\n\n{len(unrecognized)} pasted line(s) didn't match any known "
                            f"provider key format and were ignored.")
            messagebox.showinfo("Providers detected", summary)
        elif unrecognized:
            messagebox.showwarning(
                "No keys recognized",
                f"{len(unrecognized)} line(s) were pasted but none matched a known "
                "provider key format, so nothing was assigned. Continuing anyway -- use "
                "'Check API Key' in the main window if the crawl fails to start.")

    # No API key needed for the local Ollama server, so the old
    # paste-your-cloud-API-keys startup popup is skipped entirely.
    # _paste_keys_popup() is kept above (unused) in case a cloud
    # provider is ever wired back in.

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            global _GUI_ROOT
            _GUI_ROOT = self  # lets manual-paste popups triggered from the
                               # background crawler thread build safely here
            self.title("Deepship — College Financial Aid Crawler")
            self.configure(bg=BG_DARK)
            try:
                self.state("zoomed")
            except tk.TclError:
                try:
                    self.attributes("-zoomed", True)
                except tk.TclError:
                    w, h = self.winfo_screenwidth(), self.winfo_screenheight()
                    self.geometry(f"{w}x{h}+0+0")

            self._setup_style()

            self.msg_queue = queue.Queue()
            self.crawler = None
            self._running = False
            self._start_time = None
            self._stats = dict(colleges_done=0, colleges_total=0, links=0, candidates=0,
                                pages=0, scholarships=0, groq_calls=0)
            self._current_domain = ""
            self._results = []  # flat list of {college, url, **entry} for the results tree

            self.domains_var = tk.StringVar()
            self.fetch_n_var = tk.IntVar(value=5)
            self.max_candidates_var = tk.IntVar(value=DEFAULT_MAX_CANDIDATES)
            self.model_var = tk.StringVar(value=OLLAMA_MODEL)

            self._build_widgets()
            self._domains_text.insert("1.0", "\n".join(DEFAULT_SAMPLE_COLLEGES))
            self.after(100, self._poll_events)
            self.after(500, self._tick_elapsed)

        # ---------------- dark theme ----------------

        def _setup_style(self):
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure(".", background=BG_DARK, foreground=FG_BRIGHT,
                             fieldbackground=BG_PANEL2, bordercolor=BORDER,
                             darkcolor=BG_PANEL2, lightcolor=BG_PANEL2)
            style.configure("TFrame", background=BG_DARK)
            style.configure("TLabel", background=BG_DARK, foreground=FG_BRIGHT)
            style.configure("TLabelframe", background=BG_DARK, foreground=ACCENT_CYAN, bordercolor=BORDER)
            style.configure("TLabelframe.Label", background=BG_DARK, foreground=ACCENT_CYAN,
                             font=("TkDefaultFont", 9, "bold"))
            style.configure("TButton", background=BG_PANEL2, foreground=FG_BRIGHT, bordercolor=BORDER)
            style.map("TButton", background=[("active", "#242832"), ("disabled", BG_PANEL)])
            style.configure("Start.TButton", foreground=GREEN)
            style.configure("Stop.TButton", foreground=RED)
            style.configure("TEntry", fieldbackground=BG_PANEL2, foreground=FG_BRIGHT,
                             insertcolor=FG_BRIGHT, bordercolor=BORDER)
            style.configure("TSpinbox", fieldbackground=BG_PANEL2, foreground=FG_BRIGHT, bordercolor=BORDER)
            style.configure("TScrollbar", background=BG_PANEL2, troughcolor=BG_DARK, bordercolor=BORDER)
            style.configure("Treeview", background=BG_DARK, fieldbackground=BG_DARK, foreground=FG_BRIGHT,
                             bordercolor=BORDER)
            style.configure("Treeview.Heading", background=BG_PANEL2, foreground=ACCENT_CYAN)
            style.map("Treeview", background=[("selected", "#243447")])

        # ---------------- layout ----------------

        def _build_widgets(self):
            # Everything above the pipeline/content split lives in ONE
            # compact horizontal strip now instead of three stacked rows --
            # domains+buttons+options on the left, controls/provider/stats
            # sideways to the right, all in a single slim bar.
            top = tk.Frame(self, bg=BG_DARK, padx=8, pady=3)
            top.pack(fill="x")

            tk.Label(top, text="Colleges:", bg=BG_DARK, fg=ACCENT_PURPLE,
                     font=("TkDefaultFont", 8, "bold")).grid(row=0, column=0, sticky="nw")
            self._domains_text = tk.Text(top, height=2, width=34, bg=BG_PANEL2, fg=FG_BRIGHT,
                                          insertbackground=FG_BRIGHT, bd=0, highlightthickness=1,
                                          highlightbackground=BORDER, font=("Consolas", 8))
            self._domains_text.grid(row=0, column=1, padx=4, sticky="w")

            btns = tk.Frame(top, bg=BG_DARK)
            btns.grid(row=0, column=2, sticky="nw", padx=(6, 0))
            ttk.Button(btns, text="Load file…", command=self._on_load_file).pack(anchor="w")

            opts = tk.Frame(top, bg=BG_DARK)
            opts.grid(row=0, column=3, sticky="nw", padx=(14, 0))
            tk.Label(opts, text="Fetch N:", bg=BG_DARK, fg=ACCENT_AMBER, font=("TkDefaultFont", 8)).grid(
                row=0, column=0, sticky="w")
            ttk.Spinbox(opts, from_=1, to=500, textvariable=self.fetch_n_var, width=4).grid(
                row=0, column=1, sticky="w", padx=(3, 10))
            tk.Label(opts, text="Candidates/college:", bg=BG_DARK, fg=ACCENT_AMBER, font=("TkDefaultFont", 8)).grid(
                row=0, column=2, sticky="w")
            ttk.Spinbox(opts, from_=1, to=10, textvariable=self.max_candidates_var, width=4).grid(
                row=0, column=3, sticky="w", padx=(3, 10))
            tk.Label(opts, text="Model override:", bg=BG_DARK, fg=ACCENT_AMBER, font=("TkDefaultFont", 8)).grid(
                row=0, column=4, sticky="w")
            ttk.Entry(opts, textvariable=self.model_var, width=18).grid(
                row=0, column=5, sticky="w", padx=(3, 0))
            tk.Label(opts, text="(blank = each provider's own fallback list)",
                     bg=BG_DARK, fg=FG_DIM, font=("TkDefaultFont", 7)).grid(
                row=1, column=0, columnspan=6, sticky="w")

            def _sep(parent):
                tk.Frame(parent, width=1, bg=BORDER).pack(side="left", fill="y", padx=8, pady=2)

            ctrl = tk.Frame(self, bg=BG_PANEL, padx=8, pady=4, highlightthickness=1, highlightbackground=BORDER)
            ctrl.pack(fill="x", padx=8, pady=(3, 0))

            self.start_btn = ttk.Button(ctrl, text="▶ Start", style="Start.TButton",
                                         command=self._on_start_clicked)
            self.start_btn.pack(side="left")
            self.stop_btn = ttk.Button(ctrl, text="■ Stop", style="Stop.TButton",
                                        command=self._on_stop_clicked, state="disabled")
            self.stop_btn.pack(side="left", padx=3)
            ttk.Button(ctrl, text="🔑 Check Key", command=self._on_check_key_clicked).pack(side="left", padx=3)
            ttk.Button(ctrl, text="🔍 Preview Data", command=self._on_preview_data).pack(side="left", padx=3)
            _sep(ctrl)
            self.pipe_status = tk.Label(ctrl, text="Ready", bg=BG_PANEL, fg=FG_BRIGHT, font=("TkDefaultFont", 9))
            self.pipe_status.pack(side="left")
            _sep(ctrl)
            tk.Label(ctrl, text="Using:", bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8)).pack(side="left")
            self.provider_status_var = tk.StringVar(value="—")
            tk.Label(ctrl, textvariable=self.provider_status_var, bg=BG_PANEL, fg=ACCENT_AMBER,
                     font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(4, 0))
            _sep(ctrl)

            self._build_stats_bar(ctrl, _sep)

            # Top half: the pipeline boxes (big, live text per stage).
            # Bottom half: everything else, rearranged to fit underneath.
            # A vertical PanedWindow keeps the split roughly 50/50 but still
            # user-draggable.
            main_pane = tk.PanedWindow(self, orient="vertical", sashwidth=6,
                                        bg=BG_DARK, sashrelief="flat", bd=0)
            main_pane.pack(fill="both", expand=True, padx=8, pady=(4, 8))

            pipeline_pane = tk.Frame(main_pane, bg=BG_DARK)
            content_pane = tk.Frame(main_pane, bg=BG_DARK)
            main_pane.add(pipeline_pane, stretch="always")
            main_pane.add(content_pane, stretch="always")

            self._build_stage_bar(pipeline_pane)

            outer = tk.PanedWindow(content_pane, orient="horizontal", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            outer.pack(fill="both", expand=True)

            left_col = tk.PanedWindow(outer, orient="vertical", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            right_col = tk.PanedWindow(outer, orient="vertical", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            outer.add(left_col, stretch="always")
            outer.add(right_col, stretch="always")

            site_frame = tk.Frame(left_col, bg=BG_DARK)
            links_frame = tk.Frame(left_col, bg=BG_DARK)
            log_frame = tk.Frame(left_col, bg=BG_DARK)
            left_col.add(site_frame, stretch="always")
            left_col.add(links_frame, stretch="always")
            left_col.add(log_frame, stretch="always")

            text_frame = tk.Frame(right_col, bg=BG_DARK)
            groq_frame = tk.Frame(right_col, bg=BG_DARK)
            right_col.add(text_frame, stretch="always")
            right_col.add(groq_frame, stretch="always")

            self._build_site_pane(site_frame)
            self._build_links_pane(links_frame)
            self._build_provider_log_pane(log_frame)
            self._build_text_pane(text_frame)
            self._build_groq_pane(groq_frame)

            # Force the pipeline/content split to ~half-and-half once the
            # window has actually been laid out (sizes aren't known yet on
            # the same tick this widget tree gets built).
            def _split_evenly():
                main_pane.update_idletasks()
                h = main_pane.winfo_height()
                if h > 100:
                    main_pane.sash_place(0, 0, h // 2)
            self.after(150, _split_evenly)

        # ---------------- pipeline stage tracker (full width, top) ----------------

        def _build_stage_bar(self, parent):
            tk.Label(parent, text="PIPELINE — live text as each pass processes it",
                     bg=BG_DARK, fg=FG_DIM, font=("TkDefaultFont", 9, "bold")).pack(
                anchor="w", padx=10, pady=(4, 2))

            row = tk.Frame(parent, bg=BG_DARK)
            row.pack(fill="both", expand=True, padx=8, pady=(0, 6))

            self._stage_boxes = {}
            ncols = len(VISIBLE_STAGE_ORDER) * 2 - 1
            for col in range(ncols):
                row.grid_columnconfigure(col, weight=(1 if col % 2 == 0 else 0))
            row.grid_rowconfigure(0, weight=1)

            for i, stage_id in enumerate(VISIBLE_STAGE_ORDER):
                meta = STAGE_META[stage_id]
                col = i * 2

                cell = tk.Frame(row, bg=BG_DARK)
                cell.grid(row=0, column=col, sticky="nsew", padx=3)
                cell.grid_rowconfigure(1, weight=1)
                cell.grid_columnconfigure(0, weight=1)

                top_bar = tk.Frame(cell, height=6, bg=BORDER)
                top_bar.grid(row=0, column=0, sticky="ew")

                body = tk.Frame(cell, bg=BG_PANEL, highlightthickness=2, highlightbackground=BORDER)
                body.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
                body.grid_rowconfigure(1, weight=1)
                body.grid_columnconfigure(0, weight=1)

                header = tk.Frame(body, bg=BG_PANEL)
                header.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
                dot = tk.Label(header, text="○", bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 12))
                dot.pack(side="left")
                name = tk.Label(header, text=meta["label"], bg=BG_PANEL, fg=FG_DIM,
                                 font=("TkDefaultFont", 10, "bold"))
                name.pack(side="left", padx=(5, 0))
                result = tk.Label(header, text=meta["idle"], bg=BG_PANEL, fg=FG_DIM,
                                   font=("TkDefaultFont", 8), wraplength=150, justify="left")
                result.pack(side="left", padx=(6, 0))

                text_wrap = tk.Frame(body, bg=BG_PANEL)
                text_wrap.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 5))
                text_wrap.grid_rowconfigure(0, weight=1)
                text_wrap.grid_columnconfigure(0, weight=1)

                text_widget = tk.Text(text_wrap, bg=BG_PANEL2, fg=FG_DIM, bd=0, wrap="word",
                                       font=("Consolas", 8), state="disabled", highlightthickness=0)
                text_widget.grid(row=0, column=0, sticky="nsew")
                sb = ttk.Scrollbar(text_wrap, orient="vertical", command=text_widget.yview)
                sb.grid(row=0, column=1, sticky="ns")
                text_widget.configure(yscrollcommand=sb.set)

                self._stage_boxes[stage_id] = {
                    "top_bar": top_bar, "body": body, "dot": dot, "name": name,
                    "result": result, "text": text_widget, "color": meta["color"],
                }

                if i < len(VISIBLE_STAGE_ORDER) - 1:
                    tk.Label(row, text="→", bg=BG_DARK, fg=FG_DIM,
                             font=("TkDefaultFont", 16)).grid(row=0, column=col + 1)

            self._reset_stage_bar()

        def _set_stage_visual(self, stage_id, state):
            """Two real states: "active" (lit, in this stage's color) and
            everything else, which always renders as the same neutral,
            unlit look -- a stage never stays lit just because it already
            ran. Only the stage doing work right now is ever lit."""
            box = self._stage_boxes.get(stage_id)
            if not box:
                return
            color = box["color"]
            if state == "active":
                box["top_bar"].configure(bg=color)
                box["body"].configure(highlightbackground=color)
                box["dot"].configure(text="●", fg=color)
                box["name"].configure(fg=color)
            elif state == "error":
                box["top_bar"].configure(bg=RED)
                box["body"].configure(highlightbackground=RED)
                box["dot"].configure(text="✗", fg=RED)
                box["name"].configure(fg=RED)
            else:  # idle
                box["top_bar"].configure(bg=BORDER)
                box["body"].configure(highlightbackground=BORDER)
                box["dot"].configure(text="○", fg=FG_DIM)
                box["name"].configure(fg=FG_DIM)

        def _set_stage_result(self, stage_id, text):
            """Updates the small running-tally line in a stage's header."""
            box = self._stage_boxes.get(stage_id)
            if box:
                box["result"].configure(text=text)

        def _append_stage_text(self, stage_id, text):
            """Appends the actual raw text this stage just processed into
            its own box, live -- trimmed from the top once it gets long so
            a long run doesn't grow these unbounded."""
            box = self._stage_boxes.get(stage_id)
            if not box or not text:
                return
            widget = box["text"]
            widget.configure(state="normal")
            widget.insert("end", text.rstrip("\n") + "\n\n")
            line_count = int(widget.index("end-1c").split(".")[0])
            if line_count > STAGE_MAX_LINES:
                widget.delete("1.0", f"{line_count - STAGE_MAX_LINES}.0")
            widget.see("end")
            widget.configure(state="disabled")

        def _advance_stage(self, stage_id):
            """Lights up only `stage_id`; every other box drops back to its
            neutral idle look, so exactly one box is ever lit at a time --
            whichever stage is actively doing work -- rather than stages
            staying lit once they're passed."""
            if stage_id not in STAGE_ORDER:
                return
            for sid in STAGE_ORDER:
                self._set_stage_visual(sid, "active" if sid == stage_id else "idle")

        def _reset_stage_bar(self):
            for sid in STAGE_ORDER:
                self._set_stage_visual(sid, "idle")
            self._reset_stage_counts()

        def _reset_stage_counts(self):
            # Running totals for the whole scan (not per-page/per-college) --
            # these accumulate across the run so each box's tally only ever
            # grows, tracking the complete picture as it fills in.
            self._stage_counts = {sid: 0 for sid in STAGE_ORDER}
            self._pass4_flagged_count = 0
            for sid in STAGE_ORDER:
                self._set_stage_result(sid, STAGE_META[sid]["idle"])
                box = self._stage_boxes.get(sid)
                if box:
                    box["text"].configure(state="normal")
                    box["text"].delete("1.0", "end")
                    box["text"].configure(state="disabled")

        def _build_stats_bar(self, bar, _sep):
            self.stat_vars = {}
            specs = [
                ("colleges_done", "done", FG_BRIGHT),
                ("colleges_total", "queued", FG_DIM),
                ("links", "links", ACCENT_CYAN),
                ("candidates", "candidates", ACCENT_PURPLE),
                ("pages", "pages", ACCENT_CYAN),
                ("scholarships", "scholarships", GREEN),
                ("groq_calls", "calls", ACCENT_AMBER),
            ]
            for key, label, color in specs:
                cell = tk.Frame(bar, bg=BG_PANEL)
                cell.pack(side="left", padx=(0, 10))
                v = tk.StringVar(value="0")
                tk.Label(cell, textvariable=v, bg=BG_PANEL, fg=color, font=("TkDefaultFont", 10, "bold")).pack(side="left")
                tk.Label(cell, text=f" {label}", bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8)).pack(side="left")
                self.stat_vars[key] = v
            _sep(bar)
            self.elapsed_var = tk.StringVar(value="0m00s")
            cell = tk.Frame(bar, bg=BG_PANEL)
            cell.pack(side="left")
            tk.Label(cell, textvariable=self.elapsed_var, bg=BG_PANEL, fg=FG_BRIGHT, font=("TkDefaultFont", 10, "bold")).pack(side="left")
            tk.Label(cell, text=" elapsed", bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8)).pack(side="left")
            self._refresh_stats_labels()

        # ---- Pane: Provider status + live error log ----

        def _build_provider_log_pane(self, parent):
            frame = tk.LabelFrame(parent, text="Live Log (providers, retries, HTTP errors)",
                                   bg=BG_PANEL, fg=ACCENT_AMBER, padx=8, pady=8)
            frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))
            scroll = ttk.Scrollbar(frame)
            scroll.pack(side="right", fill="y")
            self.provider_log_text = tk.Text(frame, state="disabled", yscrollcommand=scroll.set,
                                              bg=BG_DARK, fg=FG_DIM, bd=0, highlightthickness=0,
                                              font=("Consolas", 8), wrap="word")
            self.provider_log_text.pack(fill="both", expand=True)
            scroll.config(command=self.provider_log_text.yview)
            self.provider_log_text.tag_configure("err", foreground=RED)
            self.provider_log_text.tag_configure("warn", foreground=ACCENT_AMBER)
            self.provider_log_text.tag_configure("info", foreground=FG_DIM)

        def _provider_log(self, text):
            """Every message call_groq()/_call_llm_once() would otherwise
            have printed to a console (provider/model tried, retries,
            timeouts, 401/403/429s, fallbacks) now lands here live instead
            of being silently dropped -- this is the fix for calls that
            were failing quietly. Also updates the "Using:" indicator by
            picking the provider/model out of the "using X/Y" or "fell
            back to X/Y" lines call_groq() logs on every successful call."""
            low = text.lower()
            if "!" in text or any(code in text for code in (" 401", " 403", " 429")) or \
               "timeout" in low or "failed" in low or "error" in low:
                tag = "err"
            elif "fell back" in low or "escalating" in low or "tapped out" in low:
                tag = "warn"
            else:
                tag = "info"
            self.provider_log_text.configure(state="normal")
            self.provider_log_text.insert("end", text.rstrip() + "\n", tag)
            line_count = int(self.provider_log_text.index("end-1c").split(".")[0])
            if line_count > MAX_LOG_LINES:
                self.provider_log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
            self.provider_log_text.see("end")
            self.provider_log_text.configure(state="disabled")

            m = re.search(r"(?:using|fell back to) ([\w.\-]+)/([\w.\-:]+)", text)
            if m:
                self.provider_status_var.set(f"{m.group(1)} / {m.group(2)}")



        # ---- Pane 1: Crawled Site ----

        def _build_site_pane(self, parent):
            frame = tk.LabelFrame(parent, text="Crawled Site", bg=BG_PANEL, fg=ACCENT_CYAN, padx=8, pady=8)
            frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))
            self.site_status_var = tk.StringVar(value="Ready — no college crawled yet.")
            tk.Label(frame, textvariable=self.site_status_var, bg=BG_PANEL, fg=FG_BRIGHT,
                     wraplength=480, justify="left", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 4))
            scroll = ttk.Scrollbar(frame)
            scroll.pack(side="right", fill="y")
            self.site_links_text = tk.Text(frame, state="disabled", yscrollcommand=scroll.set,
                                            bg=BG_DARK, fg=FG_BRIGHT, bd=0, highlightthickness=0,
                                            font=("Consolas", 9), wrap="none")
            self.site_links_text.pack(fill="both", expand=True)
            scroll.config(command=self.site_links_text.yview)

        # ---- Pane 2: Extracted Links/Pages ----

        def _build_links_pane(self, parent):
            frame = tk.LabelFrame(parent, text="Extracted Links / Pages (Groq's finaid picks)",
                                   bg=BG_PANEL, fg=ACCENT_PURPLE, padx=8, pady=8)
            frame.pack(fill="both", expand=True, padx=6, pady=(4, 0))
            columns = ("confidence", "status", "found")
            self.links_tree = ttk.Treeview(frame, columns=columns, show="tree headings", height=10)
            self.links_tree.heading("#0", text="URL")
            self.links_tree.heading("confidence", text="Confidence")
            self.links_tree.heading("status", text="Status")
            self.links_tree.heading("found", text="Scholarships")
            self.links_tree.column("#0", width=320, anchor="w")
            self.links_tree.column("confidence", width=80, anchor="center")
            self.links_tree.column("status", width=90, anchor="center")
            self.links_tree.column("found", width=90, anchor="center")
            vscroll = ttk.Scrollbar(frame, command=self.links_tree.yview)
            self.links_tree.configure(yscrollcommand=vscroll.set)
            self.links_tree.pack(side="left", fill="both", expand=True)
            vscroll.pack(side="right", fill="y")
            self._links_tree_rows = {}  # url -> item id

        # ---- Pane 3: Extracted Text ----

        def _build_text_pane(self, parent):
            frame = tk.LabelFrame(parent, text="Extracted Text (BeautifulSoup, current page)",
                                   bg=BG_PANEL, fg=ACCENT_AMBER, padx=8, pady=8)
            frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))
            self.text_status_var = tk.StringVar(value="No page fetched yet.")
            tk.Label(frame, textvariable=self.text_status_var, bg=BG_PANEL, fg=FG_DIM,
                     wraplength=480, justify="left").pack(anchor="w", pady=(0, 4))
            scroll = ttk.Scrollbar(frame)
            scroll.pack(side="right", fill="y")
            self.page_text_widget = tk.Text(frame, state="disabled", yscrollcommand=scroll.set,
                                             bg=BG_DARK, fg=FG_BRIGHT, bd=0, highlightthickness=0,
                                             font=("Consolas", 9), wrap="word")
            self.page_text_widget.pack(fill="both", expand=True)
            scroll.config(command=self.page_text_widget.yview)

        # ---- Pane 4: Groq's Response ----

        def _build_groq_pane(self, parent):
            frame = tk.LabelFrame(parent, text="Groq's Response", bg=BG_PANEL, fg=GREEN, padx=8, pady=8)
            frame.pack(fill="both", expand=True, padx=6, pady=(4, 0))

            log_scroll = ttk.Scrollbar(frame)
            log_scroll.pack(side="right", fill="y")
            self.groq_log_text = tk.Text(frame, state="disabled", yscrollcommand=log_scroll.set,
                                          bg=BG_DARK, fg=FG_BRIGHT, bd=0, highlightthickness=0,
                                          font=("Consolas", 8), wrap="word")
            self.groq_log_text.pack(fill="both", expand=True)
            log_scroll.config(command=self.groq_log_text.yview)
            self.groq_log_text.tag_configure("raw", foreground=FG_DIM)
            self.groq_log_text.tag_configure("hdr", foreground=ACCENT_CYAN)
            self.groq_log_text.tag_configure("warn", foreground=ACCENT_AMBER)
            self.groq_log_text.tag_configure("err", foreground=RED)
            self.groq_log_text.tag_configure("pass1", foreground=ACCENT_PURPLE)

            self._card_count = 0

        _TYPE_COLORS = {
            "need": ACCENT_CYAN, "merit": ACCENT_AMBER, "athletic": ACCENT_PURPLE,
            "full-ride": GREEN, "full-tuition": GREEN, "departmental": ACCENT_CYAN,
        }

        def _color_for_type(self, type_str):
            t = (type_str or "").lower()
            for key, color in self._TYPE_COLORS.items():
                if key in t:
                    return color
            return FG_DIM

        def _add_scholarship_card(self, college, type_str, summary, enriched=False):
            # The "Pass 2 rough summaries" card box was removed from the
            # layout -- this entry's already visible in the Groq log above
            # (and in "Preview Data"), so this just keeps a running count.
            self._card_count += 1

        def _clear_cards(self):
            self._card_count = 0

        # ---------------- helpers ----------------

        def _groq_log(self, text, tag=None):
            self.groq_log_text.configure(state="normal")
            if tag:
                self.groq_log_text.insert("end", text + "\n", tag)
            else:
                self.groq_log_text.insert("end", text + "\n")
            # trim old lines so this never grows unbounded over a long run
            line_count = int(self.groq_log_text.index("end-1c").split(".")[0])
            if line_count > MAX_LOG_LINES:
                self.groq_log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
            self.groq_log_text.see("end")
            self.groq_log_text.configure(state="disabled")

        def _refresh_stats_labels(self):
            for key, var in self.stat_vars.items():
                var.set(str(self._stats.get(key, 0)))

        # ---------------- control handlers ----------------

        def _on_load_file(self):
            path = filedialog.askopenfilename(title="Load college list",
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
                self._domains_text.delete("1.0", "end")
                self._domains_text.insert("1.0", "\n".join(lines))
            except OSError as exc:
                messagebox.showerror("Load failed", str(exc))

        def _parse_domains(self):
            raw = self._domains_text.get("1.0", "end")
            domains = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            n = max(1, self.fetch_n_var.get())
            return domains[:n]

        def _on_check_key_clicked(self):
            _refresh_groq_env()
            ready = bool(_configured_providers())
            title = "LLM provider ready" if ready else "No LLM provider configured"
            if ready:
                messagebox.showinfo(title, llm_provider_diagnostic())
            else:
                messagebox.showerror(title, llm_provider_diagnostic())

        def _on_preview_data(self):
            """Small popup window showing all the raw pipeline output
            collected so far (same text that's streamed into the Groq log
            pane), as a snapshot with a manual refresh button."""
            win = tk.Toplevel(self)
            win.title("Preview — Raw Output Data")
            win.configure(bg=BG_DARK)
            win.geometry("560x420")

            top = tk.Frame(win, bg=BG_DARK)
            top.pack(fill="x", padx=8, pady=(8, 4))
            tk.Label(top, text="All raw output collected so far", bg=BG_DARK, fg=FG_DIM,
                     font=("TkDefaultFont", 9, "bold")).pack(side="left")

            body = tk.Frame(win, bg=BG_DARK)
            body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            scroll = ttk.Scrollbar(body)
            scroll.pack(side="right", fill="y")
            preview_text = tk.Text(body, yscrollcommand=scroll.set, bg=BG_PANEL2, fg=FG_BRIGHT,
                                    bd=0, highlightthickness=0, font=("Consolas", 8), wrap="word")
            preview_text.pack(fill="both", expand=True)
            scroll.config(command=preview_text.yview)

            def _refresh():
                data = self.groq_log_text.get("1.0", "end-1c")
                preview_text.configure(state="normal")
                preview_text.delete("1.0", "end")
                preview_text.insert("1.0", data if data.strip() else "(nothing collected yet)")
                preview_text.configure(state="disabled")

            ttk.Button(top, text="Refresh", command=_refresh).pack(side="right")
            _refresh()

        def _on_start_clicked(self):
            if self._running:
                return
            _refresh_groq_env()  # picks up a .env edited or a var exported after this GUI launched
            reset_provider_fallback_state()  # give every provider a clean shot for this new run
            if not MANUAL_PASTE_MODE and not _configured_providers():
                messagebox.showerror("No LLM provider API keys set", llm_provider_diagnostic())
                return
            domains = self._parse_domains()
            if not domains:
                messagebox.showwarning("No colleges", "Add at least one college domain first.")
                return

            self._reset_stage_bar()
            self._running = True
            self._start_time = time.time()
            self._stats = dict(colleges_done=0, colleges_total=len(domains), links=0, candidates=0,
                                pages=0, scholarships=0, groq_calls=0)
            self._refresh_stats_labels()
            self._results = []
            self._clear_cards()
            for item in self.links_tree.get_children():
                self.links_tree.delete(item)
            self._links_tree_rows = {}
            self.site_status_var.set("Starting…")
            self.site_links_text.configure(state="normal")
            self.site_links_text.delete("1.0", "end")
            self.site_links_text.configure(state="disabled")
            self.text_status_var.set("No page fetched yet.")
            self.page_text_widget.configure(state="normal")
            self.page_text_widget.delete("1.0", "end")
            self.page_text_widget.configure(state="disabled")
            self.provider_status_var.set("—")
            self.provider_log_text.configure(state="normal")
            self.provider_log_text.delete("1.0", "end")
            self.provider_log_text.configure(state="disabled")

            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.pipe_status.configure(text=f"Running… ({len(domains)} college(s) queued)")

            self.crawler = DeepshipCrawler(
                on_event=self._on_event_from_worker,
                max_candidates=max(1, self.max_candidates_var.get()),
                model=self.model_var.get().strip() or None,
            )
            self.crawler.start_batch(domains)

        def _on_stop_clicked(self):
            if self.crawler:
                self.crawler.stop()
            self.pipe_status.configure(text="Stopping… (finishing current step)")
            self.stop_btn.configure(state="disabled")

        # ---------------- worker -> GUI event bridge ----------------

        def _on_event_from_worker(self, event, **kw):
            self.msg_queue.put((event, kw))

        def _poll_events(self):
            try:
                while True:
                    event, kw = self.msg_queue.get_nowait()
                    self._handle_event(event, kw)
            except queue.Empty:
                pass
            self.after(100, self._poll_events)

        def _tick_elapsed(self):
            if self._running and self._start_time:
                elapsed = int(time.time() - self._start_time)
                self.elapsed_var.set(f"{elapsed // 60}m{elapsed % 60:02d}s")
            self.after(500, self._tick_elapsed)

        # ---------------- event handling ----------------

        def _handle_event(self, event, kw):
            domain = kw.get("domain", "")
            url = kw.get("url", "")
            ts = datetime.now().strftime("%H:%M:%S")

            if event == "run_start":
                self._stats["colleges_total"] = kw.get("total", 0)
                self._refresh_stats_labels()

            elif event == "log_line":
                self._provider_log(kw.get("text", ""))

            elif event == "college_start":
                self._advance_stage("crawl")
                self._current_domain = domain
                idx, total = kw.get("index"), kw.get("total")
                progress = f" ({idx + 1}/{total})" if idx is not None and total else ""
                self.pipe_status.configure(text=f"Crawling {domain}{progress}")
                self._append_stage_text("crawl", f"[{ts}] === {domain} ===\nhomepage: {kw.get('homepage', '')}")
                self.site_status_var.set(f"{domain} -> {kw.get('homepage', '')}")
                self.site_links_text.configure(state="normal")
                self.site_links_text.delete("1.0", "end")
                self.site_links_text.configure(state="disabled")
                for item in self.links_tree.get_children():
                    self.links_tree.delete(item)
                self._links_tree_rows = {}
                self._groq_log(f"[{ts}] === {domain} ===", "hdr")

            elif event == "links_found":
                links = kw.get("links", [])
                self._stats["links"] += len(links)
                self._refresh_stats_labels()
                self._stage_counts["crawl"] += len(links)
                self._set_stage_result("crawl", f"{self._stage_counts['crawl']} link(s) found")
                link_lines = "\n".join(f"{(l.get('text') or '(no text)')[:50]:<50}  {l['url']}" for l in links)
                self._append_stage_text("crawl", f"[{ts}] {len(links)} link(s) found:\n{link_lines}")
                self.site_status_var.set(f"{self._current_domain} -> {kw.get('homepage', '')}  "
                                          f"({len(links)} link(s) found)")
                self.site_links_text.configure(state="normal")
                self.site_links_text.delete("1.0", "end")
                for l in links:
                    label = l.get("text") or "(no text)"
                    self.site_links_text.insert("end", f"{label[:50]:<50}  {l['url']}\n")
                self.site_links_text.configure(state="disabled")

            elif event == "link_triage_request":
                self._advance_stage("triage")
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._append_stage_text("triage", f"[{ts}] -> asking Groq to triage "
                                                    f"{kw.get('link_count', 0)} link(s) on {kw.get('homepage', '')}")
                self._groq_log(f"[{ts}] -> asking Groq to triage {kw.get('link_count', 0)} link(s) "
                                f"on {kw.get('homepage', '')}", "hdr")

            elif event == "link_triage_response":
                candidates = kw.get("candidates", [])
                self._append_stage_text("triage", f"[{ts}] <- raw reply:\n{(kw.get('raw') or '')[:2000]}")
                self._groq_log(f"[{ts}] <- Groq link-triage raw reply:", "hdr")
                self._groq_log((kw.get("raw") or "")[:2000], "raw")

            elif event == "candidates_selected":
                candidates = kw.get("candidates", [])
                self._stats["candidates"] += len(candidates)
                self._refresh_stats_labels()
                self._stage_counts["triage"] += len(candidates)
                self._set_stage_result("triage", f"{self._stage_counts['triage']} candidate page(s) selected")
                cand_lines = "\n".join(f"• {c.get('url','')} (conf: {c.get('confidence','?')})" for c in candidates)
                self._append_stage_text("triage", f"[{ts}] {len(candidates)} candidate(s) selected:\n{cand_lines}")
                for c in candidates:
                    u = c.get("url", "")
                    if u in self._links_tree_rows:
                        continue
                    item_id = self.links_tree.insert("", "end", text=u,
                                                       values=(c.get("confidence", "?"), "queued", 0))
                    self._links_tree_rows[u] = item_id

            elif event == "faq_candidate_selected":
                c = kw.get("candidate", {}) or {}
                self._groq_log(f"[{ts}]    FAQ page picked for gap-fill: {c.get('url', '')} "
                                f"({c.get('reason', '')})", "warn")

            elif event == "page_start":
                self._advance_stage("fetch")
                item_id = self._links_tree_rows.get(url)
                if item_id:
                    self.links_tree.set(item_id, "status", "fetching…")
                self.pipe_status.configure(text=f"Fetching {url}")
                self._append_stage_text("fetch", f"[{ts}] fetching {url}")

            elif event == "page_fetched":
                self._stats["pages"] += 1
                self._refresh_stats_labels()
                self._stage_counts["fetch"] += 1
                self._set_stage_result("fetch", f"{self._stage_counts['fetch']} page(s) fetched")

            elif event == "page_text_extracted":
                text = kw.get("text", "")
                preview = text[:MAX_TEXT_PREVIEW_CHARS]
                truncated_note = "" if len(text) <= MAX_TEXT_PREVIEW_CHARS else \
                    f"\n\n… [truncated for display, {len(text)} chars total]"
                self.text_status_var.set(f"{url}  ({len(text)} chars extracted)")
                self.page_text_widget.configure(state="normal")
                self.page_text_widget.delete("1.0", "end")
                self.page_text_widget.insert("1.0", preview + truncated_note)
                self.page_text_widget.configure(state="disabled")
                stage_preview = text[:3000] + ("…" if len(text) > 3000 else "")
                self._append_stage_text("fetch", f"[{ts}] extracted {len(text)} char(s) from {url}:\n{stage_preview}")

            elif event == "pass1_chunk_split":
                self._groq_log(f"[{ts}]    [pass 1] {url} split into {kw.get('chunk_count')} chunk(s) "
                                f"(ladder step: {kw.get('split_n')})", "warn")

            elif event == "pass1_chunk_request":
                self._advance_stage("pass1")
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._append_stage_text("pass1", f"[{ts}] -> chunk {kw.get('chunk_index', 0) + 1}/"
                                                   f"{kw.get('total_chunks', 1)} :: {url}")
                self._groq_log(f"[{ts}] -> [PASS 1 / loose sweep] chunk {kw.get('chunk_index', 0) + 1}/"
                                f"{kw.get('total_chunks', 1)} :: {url}", "hdr")

            elif event == "pass1_chunk_response":
                snippets = kw.get("snippets", [])
                self._stage_counts["pass1"] += len(snippets)
                self._set_stage_result("pass1", f"{self._stage_counts['pass1']} raw snippet(s) flagged")
                self._append_stage_text("pass1", f"[{ts}] <- {len(snippets)} snippet(s):\n"
                                                   f"{(kw.get('raw') or '')[:1200]}")
                self._groq_log(f"[{ts}] <- [PASS 1] found {len(snippets)} possible "
                                f"snippet(s) in this chunk:", "hdr")
                for s in snippets:
                    if not isinstance(s, dict):
                        continue
                    txt = (s.get("snippet") or "").replace("\n", " ")[:160]
                    hint = s.get("hint", "")
                    self._groq_log(f"      • ({hint}) {txt}", "pass1")
                self._groq_log((kw.get("raw") or "")[:1200], "raw")

            elif event == "pass2_request":
                self._advance_stage("pass2")
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._append_stage_text("pass2", f"[{ts}] -> batch {kw.get('batch_index', 0) + 1}/"
                                                   f"{kw.get('total_batches', 1)} :: "
                                                   f"{kw.get('snippet_count', 0)} snippet(s) :: {url}")
                self._groq_log(f"[{ts}] -> [PASS 2 / structuring] batch {kw.get('batch_index', 0) + 1}/"
                                f"{kw.get('total_batches', 1)} :: {kw.get('snippet_count', 0)} snippet(s) :: {url}",
                                "hdr")

            elif event == "pass2_response":
                entries = kw.get("entries", [])
                self._stage_counts["pass2"] += len(entries)
                self._set_stage_result("pass2", f"{self._stage_counts['pass2']} entries drafted")
                self._append_stage_text("pass2", f"[{ts}] <- {len(entries)} entr"
                                                   f"{'y' if len(entries) == 1 else 'ies'}:\n"
                                                   f"{(kw.get('raw') or '')[:2000]}")
                self._groq_log(f"[{ts}] <- [PASS 2] wrote up "
                                f"{len(entries)} program summar{'y' if len(entries) == 1 else 'ies'}:", "hdr")
                self._groq_log((kw.get("raw") or "")[:2000], "raw")
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    self._add_scholarship_card(self._current_domain, e.get("type", ""), e.get("summary", ""))

            elif event == "payload_too_large":
                pass_note = f" (pass {kw.get('pass_num')})" if kw.get("pass_num") else ""
                self._groq_log(f"[{ts}] !! payload too large{pass_note} at split={kw.get('split_n')} for {url} "
                                f"-- escalating chunk count", "warn")

            elif event == "page_done":
                snippets = kw.get("snippets", [])
                entries = kw.get("entries", [])
                self._stats["scholarships"] += len(entries)
                self._refresh_stats_labels()
                self._groq_log(f"[{ts}] === {url} done — pass 1: {len(snippets)} possible snippet(s), "
                                f"pass 2: {len(entries)} structured entr{'y' if len(entries) == 1 else 'ies'} ===",
                                "hdr")
                item_id = self._links_tree_rows.get(url)
                if item_id:
                    status = "error" if kw.get("error") else "done"
                    self.links_tree.set(item_id, "status", status)
                    self.links_tree.set(item_id, "found", f"{len(entries)} ({len(snippets)} raw)")

            elif event == "faq_fetch_start":
                self._advance_stage("pass3")
                self.pipe_status.configure(text=f"Fetching FAQ page {url}")
                self._append_stage_text("pass3", f"[{ts}] -> fetching FAQ page {url}")
                self._groq_log(f"[{ts}] -> [PASS 3 / FAQ gap-fill] fetching {url}", "hdr")

            elif event == "faq_fetched":
                self._append_stage_text("pass3", f"[{ts}] FAQ page extracted "
                                                   f"({kw.get('text_len', 0)} char(s)): {url}")
                self._groq_log(f"[{ts}]    FAQ page extracted ({kw.get('text_len', 0)} chars): {url}", "warn")

            elif event == "gap_fill_request":
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._append_stage_text("pass3", f"[{ts}] -> chunk {kw.get('chunk_index', 0) + 1}/"
                                                   f"{kw.get('total_chunks', 1)} :: {url}")
                self._groq_log(f"[{ts}] -> [PASS 3 / FAQ gap-fill] chunk {kw.get('chunk_index', 0) + 1}/"
                                f"{kw.get('total_chunks', 1)} :: {url}", "hdr")

            elif event == "gap_fill_response":
                entries = kw.get("entries", [])
                self._append_stage_text("pass3", f"[{ts}] <- {len(entries)} entr"
                                                   f"{'y' if len(entries) == 1 else 'ies'} so far:\n"
                                                   f"{(kw.get('raw') or '')[:1500]}")
                self._groq_log(f"[{ts}] <- [PASS 3] running total after this FAQ chunk: "
                                f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}", "hdr")
                self._groq_log((kw.get("raw") or "")[:1500], "raw")

            elif event == "gap_fill_done":
                entries = kw.get("entries", [])
                self._stage_counts["pass3"] += len(entries)
                self._set_stage_result("pass3", f"{self._stage_counts['pass3']} enriched via FAQ")
                self._groq_log(f"[{ts}] === [PASS 3] FAQ gap-fill done for {domain} — "
                                f"{len(entries)} enriched entr{'y' if len(entries) == 1 else 'ies'} ===", "hdr")
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    self._add_scholarship_card(domain, e.get("type", ""), e.get("summary", ""), enriched=True)

            elif event == "pass4_start":
                self._advance_stage("pass4")
                self.pipe_status.configure(text=f"Regex cross-check (pass 4, no Groq) for {domain}")
                self._append_stage_text("pass4", f"[{ts}] -> scanning raw page text for {domain}")
                self._groq_log(f"[{ts}] -> [PASS 4 / regex cross-check, no Groq call] scanning "
                                f"raw page text for {domain}", "hdr")

            elif event == "pass4_done":
                paragraphs = kw.get("paragraphs", [])
                flagged = kw.get("flagged", [])
                self._stage_counts["pass4"] += len(paragraphs)
                self._pass4_flagged_count += len(flagged)
                self._set_stage_result("pass4", f"{self._stage_counts['pass4']} matched · "
                                                 f"{self._pass4_flagged_count} flagged new")
                stage_lines = [f"[{ts}] {len(paragraphs)} matched, {len(flagged)} flagged new:"]
                for item in paragraphs:
                    flag_note = "  [NEW]" if item in flagged else ""
                    kws = ", ".join(item.get("matched_keywords", []))
                    snippet = item.get("paragraph", "").replace("\n", " ")[:300]
                    stage_lines.append(f"• ({kws}) [{item.get('source', '')}]{flag_note}\n    {snippet}")
                self._append_stage_text("pass4", "\n".join(stage_lines))
                self._groq_log(f"[{ts}] === [PASS 4] {domain} -- {len(paragraphs)} keyword-matched "
                                f"paragraph(s) found, {len(flagged)} flagged as possibly not reflected "
                                f"in pass 3's output ===", "hdr")
                for item in paragraphs:
                    tag = "warn" if item in flagged else "pass1"
                    flag_note = "  [POSSIBLY NEW / NOT IN PASS 3]" if item in flagged else ""
                    kws = ", ".join(item.get("matched_keywords", []))
                    self._groq_log(f"    • ({kws}) [{item.get('source', '')}]{flag_note}", tag)
                    snippet = item.get("paragraph", "").replace("\n", " ")
                    self._groq_log(f"      {snippet[:500]}", "raw")

            elif event == "college_done":
                self._stats["colleges_done"] += 1
                self._refresh_stats_labels()
                if kw.get("error"):
                    self._groq_log(f"[{ts}] ! {domain} failed: {kw.get('error')}", "err")

            elif event == "run_done":
                self._reset_stage_bar()
                self._running = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                note = " (stopped early)" if kw.get("stopped_early") else ""
                self.pipe_status.configure(
                    text=f"Done{note} — {kw.get('total_scholarships', 0)} scholarship(s) found.")
                self._groq_log(f"[{ts}] === run finished{note} — "
                                f"{kw.get('total_scholarships', 0)} scholarship(s) total ===", "hdr")

        def destroy(self):
            if self.crawler:
                self.crawler.stop()
            super().destroy()

    app = App()
    app.mainloop()


# ---------------------------------------------------------------------------
# CLI (headless, writes a JSON file — used when the script is run with args)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crawl college home pages, ask Groq to find the financial aid / "
                     "scholarships page, then extract scholarship details from it. "
                     "Run with no arguments to launch the live GUI instead."
    )
    parser.add_argument("colleges", nargs="*", help="College domains, e.g. princeton.edu harvard.edu")
    parser.add_argument("--file", help="Path to a text file with one college domain per line")
    parser.add_argument("--output", default="deepship_results.json", help="Where to write the JSON results")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                         help="How many Groq-ranked candidate links to deep-dive per college (default: %(default)s)")
    parser.add_argument("--model", default=None,
                         help="Pin the first model tried on every provider (each provider still "
                              "falls back through its own other models, then the next provider, "
                              "if that one's unavailable). Leave unset to use each provider's "
                              "built-in model list / <PROVIDER>_MODEL env var as-is.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    args = parser.parse_args()

    domains = list(args.colleges)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            domains.extend(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))

    if not domains:
        parser.error("Give at least one college domain, either as an argument or via --file "
                      "(or run with no arguments at all to launch the GUI)")

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    reset_provider_fallback_state()  # clean slate for this run
    if not MANUAL_PASTE_MODE and not _configured_providers():
        print("! No LLM provider API keys are set. Exiting.\n", file=sys.stderr)
        print(llm_provider_diagnostic(), file=sys.stderr)
        sys.exit(1)

    warm_up_model(model=args.model, log=log)

    all_results = []
    for domain in domains:
        result = process_college(domain, max_candidates=args.max_candidates, model=args.model, log=log)
        all_results.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_snippets = sum(len(p.get("pass1_snippets", [])) for r in all_results for p in r.get("pages", []))
    total_scholarships = sum(len(p.get("scholarships", [])) for r in all_results for p in r.get("pages", []))
    total_faq_enriched = sum(len(r.get("scholarships_faq_enriched", [])) for r in all_results)
    colleges_with_faq = sum(1 for r in all_results if r.get("faq_url"))
    total_pass4_paragraphs = sum(len(r.get("pass4_paragraphs", [])) for r in all_results)
    total_pass4_flagged = sum(len(r.get("pass4_flagged_new_info", [])) for r in all_results)
    print(f"\nDone. {len(domains)} college(s) processed.")
    print(f"  Pass 1 (loose sweep):    {total_snippets} possible scholarship/aid snippet(s) found.")
    print(f"  Pass 2 (structured):     {total_scholarships} scholarship entr"
          f"{'y' if total_scholarships == 1 else 'ies'} produced.")
    print(f"  Pass 3 (FAQ gap-fill):   {total_faq_enriched} enriched entr"
          f"{'y' if total_faq_enriched == 1 else 'ies'} across {colleges_with_faq} college(s) with an FAQ page found.")
    print(f"  Pass 4 (regex, no LLM):  {total_pass4_paragraphs} keyword-matched paragraph(s) found; "
          f"{total_pass4_flagged} flagged as possibly missing from pass 3's output.")
    print(f"  Full results (all passes) written to {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _run_gui()