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
    GROQ_API_KEY=gsk_...              (required)
    GROQ_MODEL=llama-3.3-70b-versatile   (optional, this is the default)

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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _refresh_groq_env():
    """Re-runs the .env loader and re-reads GROQ_API_KEY / GROQ_MODEL from
    os.environ. Safe to call repeatedly (e.g. right before Start is
    clicked in the GUI) so editing .env or exporting the var AFTER the
    script/GUI process already started doesn't require a full restart --
    _load_dotenv_if_present() only fills in keys not already in
    os.environ, so this never clobbers something already set."""
    global ENV_DEBUG, GROQ_API_KEY, GROQ_MODEL
    ENV_DEBUG = _load_dotenv_if_present()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_API_KEY_HERE")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", GROQ_MODEL)


def groq_key_diagnostic():
    """Builds a precise, actionable status string about GROQ_API_KEY --
    exactly which .env path was checked, whether that file exists, which
    keys it actually parsed out of it, and whether GROQ_API_KEY is set in
    the process's environment right now. Used both by call_groq()'s error
    and by the GUI's "Check API Key" button / start-time check, so the
    diagnosis is identical everywhere instead of a generic "it's missing.\""""
    lines = []
    if GROQ_API_KEY:
        masked = GROQ_API_KEY[:4] + "…" + GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 10 else "(short key)"
        lines.append(f"GROQ_API_KEY is set ({masked}).")
        source = "a .env file" if "GROQ_API_KEY" in ENV_DEBUG.get("keys_loaded", []) else "the shell/process environment"
        lines.append(f"Loaded from: {source}.")
        return "\n".join(lines)

    lines.append("GROQ_API_KEY is NOT currently set.")
    lines.append(f".env path checked: {ENV_DEBUG['env_path']}")
    if ENV_DEBUG["file_found"]:
        keys = ENV_DEBUG["keys_loaded"]
        lines.append(f".env file WAS found there. Keys it parsed from it: {keys or '(none)'}")
        if "GROQ_API_KEY" not in keys and "GROQ_API_KEY" in os.environ:
            lines.append("(GROQ_API_KEY IS in os.environ though -- odd state, try Check API Key again.)")
        elif not keys:
            lines.append("No KEY=VALUE lines were parsed at all -- check the file isn't empty/all-comments, "
                          "and isn't secretly named '.env.txt' (a common issue on Windows).")
        else:
            lines.append("GROQ_API_KEY specifically wasn't among them -- check for a typo in the key name, "
                          "stray quotes, or a line that doesn't look exactly like GROQ_API_KEY=gsk_...")
    else:
        lines.append(".env file was NOT found at that exact path.")
        lines.append("If you have a .env file, it needs to sit in the SAME FOLDER as this .py script "
                      "(not your terminal's current directory, and not renamed '.env.txt').")
    lines.append("")
    lines.append("If you used `export GROQ_API_KEY=...` in a terminal: that only applies to processes "
                  "launched FROM that same terminal session. Double-clicking the script, running it from "
                  "an IDE's Run button, or a new terminal tab will NOT see it -- the .env file is the more "
                  "reliable option for exactly this reason.")
    return "\n".join(lines)

USER_AGENT = "DeepshipFinAidCrawler/1.0 (college financial aid research; contact: set-your-email-here)"
REQUEST_TIMEOUT_S = 25
GROQ_TIMEOUT_S = 60
POLITE_DELAY_S = 0.75

# The chunking ladder described above: try whole-page-in-2, then 4, then 6.
SPLIT_LADDER = [2, 4, 6]

DEFAULT_MAX_CANDIDATES = 3
MAX_LINKS_SENT_TO_GROQ = 250  # trim an absurdly linky home page before triage

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


GROQ_RATE_LIMITER = GroqRateLimiter()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PayloadTooLargeError(Exception):
    """Raised when Groq (or a proxy in front of it) rejects a request for
    being too big -- HTTP 413, or a 400 whose body says as much (some
    providers report oversized-request as a 400 with a message instead of
    a clean 413)."""
    pass


class GroqCallError(Exception):
    pass


class StopRequested(Exception):
    """Raised internally to unwind out of the pipeline when the GUI's Stop
    button (or a stop_flag threading.Event) is set mid-run."""
    pass


def _check_stop(stop_flag):
    if stop_flag is not None and stop_flag.is_set():
        raise StopRequested()


# ---------------------------------------------------------------------------
# Low-level Groq call
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


def call_groq(messages, temperature=0.1, max_tokens=2000, model=None, log=None, max_retries=3):
    """Mirrors call_groq()/callGroq() shape from alting_ua.py / splicer_core.py:
    POST to GROQ_ENDPOINT, wait on the shared rate limiter first, retry on
    429/transient network errors, and raise PayloadTooLargeError distinctly
    so callers can escalate their chunking instead of just failing."""

    def _log(msg):
        if log:
            log(msg)

    if not GROQ_API_KEY:
        raise GroqCallError(f"GROQ_API_KEY isn't set.\n{groq_key_diagnostic()}")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": model or GROQ_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        GROQ_RATE_LIMITER.wait_turn(_log)
        try:
            resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
        except requests.exceptions.Timeout:
            last_exc = GroqCallError("Groq took too long to respond.")
            _log(f"  ! Groq timeout (attempt {attempt}/{max_retries})")
            continue
        except requests.exceptions.RequestException as exc:
            last_exc = GroqCallError(f"Groq request failed: {exc}")
            _log(f"  ! Groq network error (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(1.5 * attempt)
            continue

        if resp.status_code == 429:
            GROQ_RATE_LIMITER.note_rate_limited(_log)
            last_exc = GroqCallError(f"Groq 429: {resp.text[:300]}")
            time.sleep(1.0 * attempt)
            continue

        if _looks_like_payload_too_large(resp.status_code, resp.text):
            _log(f"  ! Groq says payload too large (HTTP {resp.status_code}) -- caller should clump smaller.")
            raise PayloadTooLargeError(resp.text[:300])

        if not resp.ok:
            raise GroqCallError(f"Groq error: HTTP {resp.status_code} {resp.text[:400]}")

        GROQ_RATE_LIMITER.note_success()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message", {}) or {}).get("content", "") or ""

    raise last_exc or GroqCallError("Groq call failed after retries.")


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
    "You are helping a crawler find the financial aid / scholarships page "
    "on a college website, given only its home page's links. You will be "
    "given a JSON list of {url, text} pairs scraped from the home page. "
    "Pick every link that plausibly leads to financial aid, scholarships, "
    "cost of attendance, tuition assistance, grants, or student financial "
    "services -- include a link even if it's just a hub/menu page that "
    "probably links onward to the real scholarships page. "
    "Respond ONLY with raw JSON (no prose, no markdown fences): a JSON "
    "array of objects, ranked best-guess first, each shaped exactly like "
    '{"url": "...", "reason": "short reason", "confidence": "high|medium|low"}. '
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
            item = {"url": item, "reason": "", "confidence": "medium"}
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
    result = {"college": domain, "homepage": homepage_url, "candidate_links": [], "pages": []}

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
    if on_event:
        on_event("candidates_selected", domain=domain, candidates=candidates[:max_candidates])
    if not candidates:
        _log("  Groq found no plausible financial aid / scholarships link on the home page.")
        if on_event:
            on_event("college_done", domain=domain, candidate_count=0, page_count=0, scholarship_count=0)
        return result

    _log(f"  Groq flagged {len(candidates)} candidate link(s), deep-diving top {max_candidates}:")
    for c in candidates[:max_candidates]:
        _log(f"    - [{c.get('confidence', '?')}] {c.get('url')} :: {c.get('reason', '')}")

    for candidate in candidates[:max_candidates]:
        _check_stop(stop_flag)
        url = candidate.get("url")
        if not url:
            continue
        time.sleep(POLITE_DELAY_S)
        page_record = {"url": url, "confidence": candidate.get("confidence"),
                        "reason": candidate.get("reason"), "pass1_snippets": [], "scholarships": []}
        if on_event:
            on_event("page_start", domain=domain, url=url, confidence=candidate.get("confidence"),
                      reason=candidate.get("reason"))
        try:
            html = fetch_url(url, log=_log)
            if on_event:
                on_event("page_fetched", domain=domain, url=url, html_len=len(html))
            text = extract_page_text(html, log=_log)
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

    def start_batch(self, domains):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(list(domains),), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self, domains):
        total = len(domains)
        self.on_event("run_start", total=total)
        total_scholarships = 0
        stopped_early = False
        for i, domain in enumerate(domains):
            if self._stop.is_set():
                stopped_early = True
                break
            try:
                result = process_college(
                    domain, max_candidates=self.max_candidates, model=self.model,
                    log=None, on_event=self.on_event, stop_flag=self._stop,
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

    MAX_TEXT_PREVIEW_CHARS = 20000
    MAX_LOG_LINES = 2000

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
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
            self.model_var = tk.StringVar(value=GROQ_MODEL)

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
            top = tk.Frame(self, bg=BG_DARK, padx=8, pady=6)
            top.pack(fill="x")

            tk.Label(top, text="Colleges (one domain per line):", bg=BG_DARK, fg=ACCENT_PURPLE).grid(
                row=0, column=0, sticky="nw")
            self._domains_text = tk.Text(top, height=4, width=42, bg=BG_PANEL2, fg=FG_BRIGHT,
                                          insertbackground=FG_BRIGHT, bd=0, highlightthickness=1,
                                          highlightbackground=BORDER, font=("Consolas", 9))
            self._domains_text.grid(row=0, column=1, rowspan=3, padx=6, sticky="w")

            btns = tk.Frame(top, bg=BG_DARK)
            btns.grid(row=0, column=2, rowspan=3, sticky="nw", padx=(6, 0))
            ttk.Button(btns, text="Load from file…", command=self._on_load_file).pack(anchor="w", pady=1)
            ttk.Button(btns, text="Reset to sample list", command=self._on_reset_sample).pack(anchor="w", pady=1)

            opts = tk.Frame(top, bg=BG_DARK)
            opts.grid(row=0, column=3, rowspan=3, sticky="nw", padx=(20, 0))

            tk.Label(opts, text="Fetch first N colleges:", bg=BG_DARK, fg=ACCENT_AMBER).grid(
                row=0, column=0, sticky="w")
            ttk.Spinbox(opts, from_=1, to=500, textvariable=self.fetch_n_var, width=5).grid(
                row=0, column=1, sticky="w", padx=(4, 0))

            tk.Label(opts, text="Candidate pages/college:", bg=BG_DARK, fg=ACCENT_AMBER).grid(
                row=1, column=0, sticky="w", pady=(4, 0))
            ttk.Spinbox(opts, from_=1, to=10, textvariable=self.max_candidates_var, width=5).grid(
                row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

            tk.Label(opts, text="Groq model:", bg=BG_DARK, fg=ACCENT_AMBER).grid(
                row=2, column=0, sticky="w", pady=(4, 0))
            ttk.Entry(opts, textvariable=self.model_var, width=24).grid(
                row=2, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

            ctrl = tk.Frame(self, bg=BG_DARK, padx=8)
            ctrl.pack(fill="x")
            self.start_btn = ttk.Button(ctrl, text="▶ Start Crawl", style="Start.TButton",
                                         command=self._on_start_clicked)
            self.start_btn.pack(side="left")
            self.stop_btn = ttk.Button(ctrl, text="■ Stop", style="Stop.TButton",
                                        command=self._on_stop_clicked, state="disabled")
            self.stop_btn.pack(side="left", padx=4)
            ttk.Button(ctrl, text="🔑 Check API Key", command=self._on_check_key_clicked).pack(side="left", padx=4)
            self.pipe_status = tk.Label(ctrl, text="Ready", bg=BG_DARK, fg=FG_BRIGHT, font=("TkDefaultFont", 10))
            self.pipe_status.pack(side="left", padx=16)

            self._build_stats_bar()

            outer = tk.PanedWindow(self, orient="horizontal", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            outer.pack(fill="both", expand=True, padx=8, pady=(0, 8))

            left_col = tk.PanedWindow(outer, orient="vertical", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            right_col = tk.PanedWindow(outer, orient="vertical", sashwidth=6, bg=BG_DARK, sashrelief="flat", bd=0)
            outer.add(left_col, stretch="always")
            outer.add(right_col, stretch="always")

            site_frame = tk.Frame(left_col, bg=BG_DARK)
            links_frame = tk.Frame(left_col, bg=BG_DARK)
            left_col.add(site_frame, stretch="always")
            left_col.add(links_frame, stretch="always")

            text_frame = tk.Frame(right_col, bg=BG_DARK)
            groq_frame = tk.Frame(right_col, bg=BG_DARK)
            right_col.add(text_frame, stretch="always")
            right_col.add(groq_frame, stretch="always")

            self._build_site_pane(site_frame)
            self._build_links_pane(links_frame)
            self._build_text_pane(text_frame)
            self._build_groq_pane(groq_frame)

        def _build_stats_bar(self):
            bar = tk.Frame(self, bg=BG_PANEL, padx=10, pady=6, highlightthickness=1, highlightbackground=BORDER)
            bar.pack(fill="x", padx=8, pady=(4, 0))
            self.stat_vars = {}
            specs = [
                ("colleges_done", "colleges done", FG_BRIGHT),
                ("colleges_total", "colleges queued", FG_DIM),
                ("links", "links found", ACCENT_CYAN),
                ("candidates", "candidates", ACCENT_PURPLE),
                ("pages", "pages fetched", ACCENT_CYAN),
                ("scholarships", "scholarships found", GREEN),
                ("groq_calls", "groq calls", ACCENT_AMBER),
            ]
            for key, label, color in specs:
                cell = tk.Frame(bar, bg=BG_PANEL)
                cell.pack(side="left", padx=(0, 22))
                v = tk.StringVar(value="0")
                tk.Label(cell, textvariable=v, bg=BG_PANEL, fg=color, font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
                tk.Label(cell, text=label, bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8)).pack(anchor="w")
                self.stat_vars[key] = v
            self.elapsed_var = tk.StringVar(value="0m00s")
            cell = tk.Frame(bar, bg=BG_PANEL)
            cell.pack(side="right")
            tk.Label(cell, textvariable=self.elapsed_var, bg=BG_PANEL, fg=FG_BRIGHT, font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
            tk.Label(cell, text="elapsed", bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8)).pack(anchor="w")
            self._refresh_stats_labels()

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

            groq_split = tk.PanedWindow(frame, orient="vertical", sashwidth=6, bg=BG_PANEL, sashrelief="flat", bd=0)
            groq_split.pack(fill="both", expand=True)

            log_frame = tk.Frame(groq_split, bg=BG_PANEL)
            results_frame = tk.Frame(groq_split, bg=BG_PANEL)
            groq_split.add(log_frame, stretch="always")
            groq_split.add(results_frame, stretch="always")

            log_scroll = ttk.Scrollbar(log_frame)
            log_scroll.pack(side="right", fill="y")
            self.groq_log_text = tk.Text(log_frame, state="disabled", yscrollcommand=log_scroll.set,
                                          bg=BG_DARK, fg=FG_BRIGHT, bd=0, highlightthickness=0,
                                          font=("Consolas", 8), wrap="word")
            self.groq_log_text.pack(fill="both", expand=True)
            log_scroll.config(command=self.groq_log_text.yview)
            self.groq_log_text.tag_configure("raw", foreground=FG_DIM)
            self.groq_log_text.tag_configure("hdr", foreground=ACCENT_CYAN)
            self.groq_log_text.tag_configure("warn", foreground=ACCENT_AMBER)
            self.groq_log_text.tag_configure("err", foreground=RED)
            self.groq_log_text.tag_configure("pass1", foreground=ACCENT_PURPLE)

            tk.Label(results_frame, text="Pass 2 — rough program summaries found so far "
                                          "(pass 1's raw snippets stream in the log above):",
                     bg=BG_PANEL, fg=FG_DIM, font=("TkDefaultFont", 8, "bold")).pack(anchor="w", pady=(0, 4))

            cards_outer = tk.Frame(results_frame, bg=BG_PANEL)
            cards_outer.pack(fill="both", expand=True)
            self.cards_canvas = tk.Canvas(cards_outer, bg=BG_PANEL, bd=0, highlightthickness=0)
            cscroll = ttk.Scrollbar(cards_outer, orient="vertical", command=self.cards_canvas.yview)
            self.cards_canvas.configure(yscrollcommand=cscroll.set)
            self.cards_canvas.pack(side="left", fill="both", expand=True)
            cscroll.pack(side="right", fill="y")

            self.cards_container = tk.Frame(self.cards_canvas, bg=BG_PANEL)
            self._cards_window = self.cards_canvas.create_window((0, 0), window=self.cards_container, anchor="nw")

            def _sync_scrollregion(_event=None):
                self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all"))

            def _sync_card_width(event):
                self.cards_canvas.itemconfigure(self._cards_window, width=event.width)

            self.cards_container.bind("<Configure>", _sync_scrollregion)
            self.cards_canvas.bind("<Configure>", _sync_card_width)

            def _on_mousewheel(event):
                delta = -1 if event.delta > 0 else 1
                if event.num == 5:
                    delta = 1
                elif event.num == 4:
                    delta = -1
                self.cards_canvas.yview_scroll(delta, "units")

            self.cards_canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows/macOS
            self.cards_canvas.bind_all("<Button-4>", _on_mousewheel)     # Linux scroll up
            self.cards_canvas.bind_all("<Button-5>", _on_mousewheel)     # Linux scroll down

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

        def _add_scholarship_card(self, college, type_str, summary):
            badge_color = self._color_for_type(type_str)
            card = tk.Frame(self.cards_container, bg=BG_PANEL2, highlightthickness=1,
                             highlightbackground=BORDER, padx=8, pady=6)
            card.pack(fill="x", padx=2, pady=3)

            top_row = tk.Frame(card, bg=BG_PANEL2)
            top_row.pack(fill="x")
            tk.Label(top_row, text=college, bg=BG_PANEL2, fg=FG_BRIGHT,
                     font=("TkDefaultFont", 9, "bold")).pack(side="left")
            tk.Label(top_row, text=(type_str or "unclassified"), bg=BG_PANEL2, fg=badge_color,
                     font=("TkDefaultFont", 8, "bold")).pack(side="right")

            tk.Label(card, text=summary or "(no summary text)", bg=BG_PANEL2, fg=FG_DIM,
                     wraplength=440, justify="left", font=("TkDefaultFont", 9)).pack(
                fill="x", anchor="w", pady=(3, 0))

            self._card_count += 1
            self.cards_canvas.after_idle(
                lambda: self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all")))

        def _clear_cards(self):
            for child in self.cards_container.winfo_children():
                child.destroy()
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

        def _on_reset_sample(self):
            self._domains_text.delete("1.0", "end")
            self._domains_text.insert("1.0", "\n".join(DEFAULT_SAMPLE_COLLEGES))

        def _parse_domains(self):
            raw = self._domains_text.get("1.0", "end")
            domains = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            n = max(1, self.fetch_n_var.get())
            return domains[:n]

        def _on_check_key_clicked(self):
            _refresh_groq_env()
            title = "GROQ_API_KEY found" if GROQ_API_KEY else "GROQ_API_KEY missing"
            if GROQ_API_KEY:
                messagebox.showinfo(title, groq_key_diagnostic())
            else:
                messagebox.showerror(title, groq_key_diagnostic())

        def _on_start_clicked(self):
            if self._running:
                return
            _refresh_groq_env()  # picks up a .env edited or a var exported after this GUI launched
            if not GROQ_API_KEY:
                messagebox.showerror("GROQ_API_KEY missing", groq_key_diagnostic())
                return
            domains = self._parse_domains()
            if not domains:
                messagebox.showwarning("No colleges", "Add at least one college domain first.")
                return

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

            elif event == "college_start":
                self._current_domain = domain
                idx, total = kw.get("index"), kw.get("total")
                progress = f" ({idx + 1}/{total})" if idx is not None and total else ""
                self.pipe_status.configure(text=f"Crawling {domain}{progress}")
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
                self.site_status_var.set(f"{self._current_domain} -> {kw.get('homepage', '')}  "
                                          f"({len(links)} link(s) found)")
                self.site_links_text.configure(state="normal")
                self.site_links_text.delete("1.0", "end")
                for l in links:
                    label = l.get("text") or "(no text)"
                    self.site_links_text.insert("end", f"{label[:50]:<50}  {l['url']}\n")
                self.site_links_text.configure(state="disabled")

            elif event == "link_triage_request":
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._groq_log(f"[{ts}] -> asking Groq to triage {kw.get('link_count', 0)} link(s) "
                                f"on {kw.get('homepage', '')}", "hdr")

            elif event == "link_triage_response":
                candidates = kw.get("candidates", [])
                self._groq_log(f"[{ts}] <- Groq link-triage raw reply:", "hdr")
                self._groq_log((kw.get("raw") or "")[:2000], "raw")

            elif event == "candidates_selected":
                candidates = kw.get("candidates", [])
                self._stats["candidates"] += len(candidates)
                self._refresh_stats_labels()
                for c in candidates:
                    u = c.get("url", "")
                    if u in self._links_tree_rows:
                        continue
                    item_id = self.links_tree.insert("", "end", text=u,
                                                       values=(c.get("confidence", "?"), "queued", 0))
                    self._links_tree_rows[u] = item_id

            elif event == "page_start":
                item_id = self._links_tree_rows.get(url)
                if item_id:
                    self.links_tree.set(item_id, "status", "fetching…")
                self.pipe_status.configure(text=f"Fetching {url}")

            elif event == "page_fetched":
                self._stats["pages"] += 1
                self._refresh_stats_labels()

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

            elif event == "pass1_chunk_split":
                self._groq_log(f"[{ts}]    [pass 1] {url} split into {kw.get('chunk_count')} chunk(s) "
                                f"(ladder step: {kw.get('split_n')})", "warn")

            elif event == "pass1_chunk_request":
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._groq_log(f"[{ts}] -> [PASS 1 / loose sweep] chunk {kw.get('chunk_index', 0) + 1}/"
                                f"{kw.get('total_chunks', 1)} :: {url}", "hdr")

            elif event == "pass1_chunk_response":
                snippets = kw.get("snippets", [])
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
                self._stats["groq_calls"] += 1
                self._refresh_stats_labels()
                self._groq_log(f"[{ts}] -> [PASS 2 / structuring] batch {kw.get('batch_index', 0) + 1}/"
                                f"{kw.get('total_batches', 1)} :: {kw.get('snippet_count', 0)} snippet(s) :: {url}",
                                "hdr")

            elif event == "pass2_response":
                entries = kw.get("entries", [])
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

            elif event == "college_done":
                self._stats["colleges_done"] += 1
                self._refresh_stats_labels()
                if kw.get("error"):
                    self._groq_log(f"[{ts}] ! {domain} failed: {kw.get('error')}", "err")

            elif event == "run_done":
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
    parser.add_argument("--model", default=None, help="Override GROQ_MODEL for this run")
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
            print(msg)

    if not GROQ_API_KEY:
        print("! GROQ_API_KEY isn't set (env var or .env file next to this script). Exiting.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for domain in domains:
        result = process_college(domain, max_candidates=args.max_candidates, model=args.model, log=log)
        all_results.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_snippets = sum(len(p.get("pass1_snippets", [])) for r in all_results for p in r.get("pages", []))
    total_scholarships = sum(len(p.get("scholarships", [])) for r in all_results for p in r.get("pages", []))
    print(f"\nDone. {len(domains)} college(s) processed.")
    print(f"  Pass 1 (loose sweep):  {total_snippets} possible scholarship/aid snippet(s) found.")
    print(f"  Pass 2 (structured):   {total_scholarships} scholarship entr"
          f"{'y' if total_scholarships == 1 else 'ies'} produced.")
    print(f"  Full results (both passes) written to {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _run_gui()