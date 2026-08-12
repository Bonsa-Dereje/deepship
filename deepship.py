#!/usr/bin/env python3
"""
deepship.py  (v3 — two independent pipelines, run one at a time)
--------------------------------------------------------------------
Pulls financial-aid / scholarship facts out of ONE college site at a time,
talking to a local Ollama server running qwen2.5:1.5b over Tailscale. GUI is
plain HTML/CSS/JS served by a tiny built-in Python web server (no Tkinter,
no external JS/CSS frameworks) with live updates pushed over Server-Sent
Events (SSE).

Two independent sections, each with its own Run/Stop button so you can run
them one at a time against the same site:

  SNIPE (top section)
    Crawl -> triage -> fetch the financial-aid page -> walk it into whole
    PARAGRAPHS (not sentences) -> for each paragraph, ask qwen to hand back,
    VERBATIM, any sentences in that paragraph that talk about scholarships
    -- no summarizing, no paraphrasing, no JSON facts, just the original
    text back. The Python side then regex-matches each returned sentence
    against the original paragraph to find exactly where it sits, and the
    UI highlights it in place inside the full paragraph.

  FILTER PASS (bottom section)
    The original pipeline: same crawl/triage/fetch, then the page is walked
    SENTENCE by sentence, and each sentence is sent to qwen one at a time
    with a trimmed window of what's known so far, asking for a small JSON
    delta to merge into a structured knowledge object (overview, amounts,
    deadlines, how to apply, notes, etc).

Requirements:
    pip install requests beautifulsoup4

Config (env vars, or a .env file sitting next to this script):
    OLLAMA_BASE_URL=https://offr.tail05ae98.ts.net   (your Tailscale Ollama host)
    OLLAMA_MODEL=qwen2.5:1.5b

Usage:
    python3 deepship.py                    # launches the web GUI (opens browser)
    python3 deepship.py --port 9000        # same, on a different port
    python3 deepship.py princeton.edu      # headless CLI (filter-pass), writes a JSON file
    python3 deepship.py princeton.edu --output princeton_finaid.json
"""

import argparse
import json
import os
import re
import sys
import time
import threading
import queue
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# .env loader (KEY=VALUE lines, no deps, never overrides a real env var)
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
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (deepship/3.0 crawler)"}
REQUEST_TIMEOUT = 20        # page fetches
MODEL_TIMEOUT = 120         # local model round-trips can be slow -- be patient
MAX_RETRIES = 3             # network/model calls get this many attempts total
RETRY_BACKOFF = 2.0         # seconds, doubles each retry (2, 4, 8...)


# ---------------------------------------------------------------------------
# Knowledge JSON -- the thing that gets filled up and re-ingested every call
# (used by the FILTER PASS pipeline only)
# ---------------------------------------------------------------------------

def empty_knowledge(college):
    return {
        "college": college,
        "financial_aid_overview": [],   # short facts about how aid works there
        "scholarships": [],             # {"name":.., "amount":.., "eligibility":.., "deadline":..} or plain strings
        "amounts": [],                  # any specific dollar figures / % of need met, etc.
        "deadlines": [],
        "how_to_apply": [],
        "notes": [],                    # anything relevant that doesn't fit above
    }


def trim_knowledge_for_prompt(knowledge, max_items=5, max_chars=1600):
    """What we actually SEND to the model each call -- a small window, not
    the whole history, so requests stay minimal even as knowledge grows."""
    trimmed = {}
    for k, v in knowledge.items():
        trimmed[k] = v[-max_items:] if isinstance(v, list) else v
    if len(json.dumps(trimmed, ensure_ascii=False)) > max_chars:
        for k, v in trimmed.items():
            if isinstance(v, list):
                trimmed[k] = v[-3:]
    return trimmed


def merge_knowledge(canonical, model_delta):
    """Additively merges whatever the model returned into the FULL
    canonical knowledge JSON. Nothing is ever dropped just because the
    model only saw a trimmed window -- lists only grow, duplicates skipped."""
    if not isinstance(model_delta, dict):
        return canonical
    merged = json.loads(json.dumps(canonical))  # cheap deep copy
    for k, v in model_delta.items():
        if isinstance(v, list):
            existing = merged.get(k) if isinstance(merged.get(k), list) else []
            for item in v:
                if item not in existing:
                    existing.append(item)
            merged[k] = existing
        elif isinstance(v, dict):
            existing = merged.get(k) if isinstance(merged.get(k), dict) else {}
            existing.update(v)
            merged[k] = existing
        elif v not in (None, ""):
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------

def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _with_retries(fn, what, section=None, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
    """Runs fn() with retries on network errors (timeouts, connection resets,
    etc). Publishes a 'retry' event to the bus between attempts so the UI can
    show it happening, instead of just dying on the first hiccup."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except requests.RequestException as e:
            last_exc = e
            if attempt >= max_retries:
                break
            wait = backoff * (2 ** (attempt - 1))
            try:
                bus.publish("retry", section=section, what=what, attempt=attempt, max_retries=max_retries,
                            wait=round(wait, 1), message=str(e))
            except NameError:
                pass  # bus not defined yet (shouldn't happen once module is loaded)
            time.sleep(wait)
    raise last_exc


def _call_ollama(messages, timeout=MODEL_TIMEOUT, section=None, model=None):
    payload = {
        "model": model or OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0},
    }

    def attempt():
        t0 = time.time()
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        elapsed = time.time() - t0
        content = (resp.json().get("message") or {}).get("content", "")
        return _strip_fences(content), elapsed

    return _with_retries(attempt, "model round-trip", section=section)


# --- FILTER PASS: one sentence at a time, minimal structured-JSON delta ----

ASK_SYSTEM_PROMPT = (
    "You track financial-aid and scholarship facts for ONE college, one "
    "sentence at a time. You will get: the JSON of what's known so far "
    "(a trimmed recent window, not everything), a context tag describing "
    "where this sentence came from on the page, and exactly one new "
    "sentence. If the sentence is tagged as a bullet point, treat it as "
    "one line item -- if it says it's in the same list as previous "
    "bullet(s), it's likely a sibling fact (another scholarship, another "
    "amount, another rule) rather than a continuation of a sentence. "
    "Reply with ONLY a small JSON object containing whatever NEW fields/"
    "items you can add (same shape as the 'known' JSON: financial_aid_"
    "overview, scholarships, amounts, deadlines, how_to_apply, notes -- "
    "all lists of short strings except scholarships, which can be short "
    "strings or {name, amount, eligibility, deadline} objects). If the "
    "sentence has nothing relevant, reply with {}. No markdown, no "
    "commentary, no repeating things you weren't given -- keep it small."
)


def build_ask_messages(knowledge, sentence, context_tag):
    """Builds the exact messages list that will be sent to the model for one
    sentence -- pulled out on its own so the caller can show the full prompt
    in the UI *before* the round-trip completes, not just log it after."""
    trimmed_known = trim_knowledge_for_prompt(knowledge)
    user_payload = {"known": trimmed_known, "context": context_tag, "sentence": sentence}
    return [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def format_prompt_for_display(messages):
    """Renders a messages list as the full, literal text that goes to Ollama
    -- for the UI's 'prompt sent to the model' panel."""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def ask_qwen_for_sentence(messages, section="filter", model=None):
    raw, elapsed = _call_ollama(messages, section=section, model=model)
    try:
        delta = json.loads(raw)
        if not isinstance(delta, dict):
            delta = {}
    except Exception:
        delta = {}
    return delta, elapsed, raw


def ask_qwen_pick_link(links, section=None, model=None):
    """Only used if the regex triage finds nothing. Minimal payload: index
    + short anchor text only, capped to 40 links."""
    trimmed = [{"i": i, "t": (l["text"] or l["url"])[:60]} for i, l in enumerate(links[:40])]
    messages = [
        {"role": "system", "content": (
            "Pick the single link most likely to be the financial aid or "
            "scholarships page. Reply with ONLY {\"i\": <index>}."
        )},
        {"role": "user", "content": json.dumps(trimmed, ensure_ascii=False)},
    ]
    try:
        raw, _ = _call_ollama(messages, timeout=60, section=section, model=model)
        idx = json.loads(raw).get("i")
        return idx if isinstance(idx, int) else None
    except Exception:
        return None


# --- SNIPE: paragraph shown whole, but classified sentence by sentence ----

SNIPE_SYSTEM_PROMPT = (
    "You will be given ONE sentence copied exactly from a college's "
    "financial-aid webpage, plus a context tag telling you where it came "
    "from on the page. Answer strictly based on THIS sentence alone -- "
    "do not assume anything that isn't stated in it. Answer two yes/no "
    "questions: (1) \"financial_aid\" -- does this sentence talk about "
    "financial aid, a scholarship, or a grant at all (whether it says one "
    "is offered, not offered, who qualifies, deadlines, how to apply, "
    "etc.)? (2) \"amount_mentioned\" -- does this sentence state a "
    "specific amount of aid provided (a dollar figure, a percentage of "
    "need/cost met, or similar)? Also give a short \"reason\", under 20 "
    "words, in plain English, explaining your two answers using only "
    "what's in the sentence. Reply with ONLY this JSON shape: "
    "{\"financial_aid\": true|false, \"amount_mentioned\": true|false, "
    "\"reason\": \"...\"}. No markdown, no commentary outside that JSON."
)


def build_snipe_sentence_messages(sentence_text, context_tag):
    user_payload = {"context": context_tag, "sentence": sentence_text}
    return [
        {"role": "system", "content": SNIPE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def ask_qwen_classify_sentence(messages, section="snipe", model=None):
    raw, elapsed = _call_ollama(messages, section=section, model=model)
    financial_aid, amount_mentioned, reason = False, False, ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            financial_aid = bool(parsed.get("financial_aid"))
            amount_mentioned = bool(parsed.get("amount_mentioned"))
            r = parsed.get("reason")
            reason = r.strip() if isinstance(r, str) else ""
    except Exception:
        pass
    return financial_aid, amount_mentioned, reason, elapsed, raw


def snipe_category(financial_aid, amount_mentioned):
    if financial_aid and amount_mentioned:
        return "both"
    if financial_aid:
        return "aid"
    if amount_mentioned:
        return "amount"
    return None


# --- SNIPE: detail-ranking pass -- runs after a paragraph's sentences are
# all classified, comparing each pair of CONSECUTIVE highlighted sentences
# (in the order they appear in the paragraph) and asking qwen which one
# carries more concrete detail. Each win is tallied, then every highlighted
# sentence in the paragraph is graded from 0 (least detail) up to N-1
# (most detail), where N is how many highlighted sentences were in that
# paragraph -- so the grade scale is always "0 to the number of sentences".

DETAIL_COMPARE_SYSTEM_PROMPT = (
    "You will be given two sentences, A and B, both copied verbatim from "
    "the same college financial-aid webpage, plus the context they came "
    "from. Decide which sentence carries MORE concrete detail about "
    "financial aid or scholarships -- specific dollar amounts, named "
    "programs, eligibility rules, deadlines, or procedures -- rather than "
    "being vague or general. Reply with ONLY this JSON shape: "
    "{\"more_detailed\": \"A\"|\"B\", \"reason\": \"...\"} where reason is "
    "under 20 words. No markdown, no commentary outside that JSON."
)


def build_detail_compare_messages(sentence_a, sentence_b, context_tag):
    user_payload = {"context": context_tag, "A": sentence_a, "B": sentence_b}
    return [
        {"role": "system", "content": DETAIL_COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def ask_qwen_compare_detail(messages, section="snipe", model=None):
    raw, elapsed = _call_ollama(messages, section=section, model=model)
    winner, reason = "A", ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            w = parsed.get("more_detailed")
            if w in ("A", "B"):
                winner = w
            r = parsed.get("reason")
            reason = r.strip() if isinstance(r, str) else ""
    except Exception:
        pass
    return winner, reason, elapsed, raw


def rank_block_hits_by_detail(block_hits, context_tag, section, model, block_index, blocks_total):
    """Mutates each dict in block_hits in place, adding "detail_grade" (0 ..
    N-1) and "detail_max" (N-1).

    Runs a "running champion" tournament, one round at a time:
      - Round starts with every not-yet-graded sentence still in the pool,
        in the order they appear in the paragraph.
      - The first sentence in the pool is the "champion". It's compared
        (the "WHICH" node) against the next sentence in the pool; whichever
        the model says has more detail becomes/stays the champion, and the
        loser is dropped for the rest of THIS round -- the node loops back
        and the champion is compared against the next sentence down the
        list, one by one, until the pool is exhausted.
      - Whoever is champion at the end of the round gets the next grade
        down from the top (first round winner = N-1, i.e. "number one").
      - That winner is removed from the pool and the whole thing iterates
        again on what's left, until one sentence remains (which
        automatically gets grade 0).
    """
    n = len(block_hits)
    if n == 0:
        return

    remaining = list(range(n))  # indices into block_hits, still in the running
    next_grade = n - 1          # first round winner gets the highest grade
    round_num = 0

    while remaining:
        if len(remaining) == 1:
            idx = remaining[0]
            block_hits[idx]["detail_grade"] = next_grade
            block_hits[idx]["detail_max"] = n - 1
            bus.publish("detail_round_winner", section=section, index=block_index, total=blocks_total,
                        round=round_num, winner_idx=idx, sentence=block_hits[idx]["sentence"],
                        grade=next_grade, remaining_count=0)
            break

        bus.publish("detail_round_start", section=section, index=block_index, total=blocks_total,
                    round=round_num,
                    pool=[{"idx": k, "sentence": block_hits[k]["sentence"]} for k in remaining])

        champion = remaining[0]
        for challenger in remaining[1:]:
            a, b = block_hits[champion], block_hits[challenger]
            messages = build_detail_compare_messages(a["sentence"], b["sentence"], context_tag)
            bus.publish("detail_bout_start", section=section, index=block_index, total=blocks_total,
                        round=round_num, champion_idx=champion, challenger_idx=challenger,
                        sentence_a=a["sentence"], sentence_b=b["sentence"])

            winner, reason, elapsed, raw = ask_qwen_compare_detail(messages, section=section, model=model)
            winner_idx = champion if winner == "A" else challenger
            loser_idx = challenger if winner_idx == champion else champion

            bus.publish("detail_bout_done", section=section, index=block_index, total=blocks_total,
                        round=round_num, champion_idx=champion, challenger_idx=challenger,
                        winner_idx=winner_idx, loser_idx=loser_idx, reason=reason, elapsed=round(elapsed, 2),
                        sentence_a=a["sentence"], sentence_b=b["sentence"])
            champion = winner_idx

        block_hits[champion]["detail_grade"] = next_grade
        block_hits[champion]["detail_max"] = n - 1
        remaining.remove(champion)
        bus.publish("detail_round_winner", section=section, index=block_index, total=blocks_total,
                    round=round_num, winner_idx=champion, sentence=block_hits[champion]["sentence"],
                    grade=next_grade, remaining_count=len(remaining))
        next_grade -= 1
        round_num += 1

    ranked_payload = [{"sentence": h["sentence"], "grade": h["detail_grade"], "category": h["category"],
                        "start": h.get("start"), "end": h.get("end")} for h in block_hits]
    top = max(block_hits, key=lambda h: h["detail_grade"])
    bus.publish("para_ranked", section=section, index=block_index, total=blocks_total, count=n,
                ranked=ranked_payload,
                top={"sentence": top["sentence"], "grade": top["detail_grade"], "category": top["category"],
                     "start": top.get("start"), "end": top.get("end")})


def split_sentences_with_spans(text):
    """Splits `text` into sentences AND returns each one's exact (start, end)
    offset within `text`, so the UI can highlight precisely without any
    regex/fuzzy matching against model output -- the model never has to
    copy anything back, it only judges one sentence we already located."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text)]
    spans, pos = [], 0
    for part in parts:
        if not part:
            continue
        idx = text.find(part, pos)
        if idx == -1:
            idx = text.find(part)
        if idx == -1:
            continue
        start, end = idx, idx + len(part)
        spans.append((part, start, end))
        pos = end
    return spans



# ---------------------------------------------------------------------------
# Crawling / extraction (shared by both pipelines)
# ---------------------------------------------------------------------------

def fetch_html(url, section=None):
    def attempt():
        r = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text

    return _with_retries(attempt, f"fetch {url}", section=section)


def extract_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        links.append({"url": full, "text": a.get_text(" ", strip=True)})
    return links


FINAID_KEYWORDS = re.compile(r"financ|scholarship|aid|tuition|fafsa|grant|afford|cost", re.I)


def guess_finaid_link(links, host):
    scored = []
    for l in links:
        if host not in urlparse(l["url"]).netloc:
            continue
        score = 2 * len(FINAID_KEYWORDS.findall(l["url"])) + len(FINAID_KEYWORDS.findall(l["text"]))
        if score > 0:
            scored.append((score, l))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored else None


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]


def _walk_blocks_raw(html):
    """Shared DOM walk: yields (element_name, text, context_tag) in document
    order for headings/paragraphs/list-items, tagging bullets that belong to
    the same list run just like the original single-pass walker did."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    bullet_pos = {}  # id(parent list) -> how many bullets seen so far in it

    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        if el.name.startswith("h"):
            yield el.name, txt, "heading"
        elif el.name == "li":
            parent_list = el.find_parent(["ul", "ol"])
            key = id(parent_list) if parent_list else id(el)
            bullet_pos[key] = bullet_pos.get(key, 0) + 1
            pos = bullet_pos[key]
            tag = ("bullet-point (start of a list)" if pos == 1 else
                   f"bullet-point (#{pos} in the same list as the previous bullet(s))")
            yield el.name, txt, tag
        else:
            yield el.name, txt, "paragraph"


def walk_items(html):
    """FILTER PASS granularity: flattens the page into single sentences,
    each tagged with structural context."""
    items = []
    for _el_name, txt, context in _walk_blocks_raw(html):
        for s in split_sentences(txt):
            items.append({"text": s, "context": context})
    return items


def walk_blocks(html):
    """SNIPE granularity: the SAME structural walk, but each block (a whole
    <p>, a whole <li>, a whole heading) is kept intact -- an entire
    paragraph handed to the model in one piece, never split into sentences."""
    items = []
    for _el_name, txt, context in _walk_blocks_raw(html):
        items.append({"text": txt, "context": context})
    return items


# ---------------------------------------------------------------------------
# Event bus -> Server-Sent Events
# ---------------------------------------------------------------------------

class EventBus:
    def __init__(self):
        self._subs = []
        self._lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event, **data):
        payload = json.dumps({"event": event, "ts": datetime.now().strftime("%H:%M:%S"), **data},
                              ensure_ascii=False, default=str)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            q.put(payload)


bus = EventBus()

# Each section runs fully independently -- its own running/stop_flag -- so
# the UI can offer one Run button per section and the person can fire them
# off one at a time (or, technically, in parallel, though the UI is built
# for "one at a time").
RUN_STATE = {
    "snipe": {"running": False, "stop_flag": False},
    "filter": {"running": False, "stop_flag": False},
}


def _find_candidate_page(section, site, model=None):
    """Crawl + triage, shared by both pipelines. Returns (base_url, host,
    candidate_link) or None (having already published an error event)."""
    site = (site or "").strip()
    if not site:
        bus.publish("error", section=section, message="No site given.")
        return None
    base_url = site if site.startswith("http") else f"https://{site}"
    host = urlparse(base_url).netloc or site

    bus.publish("crawl_start", section=section, site=site)
    html = fetch_html(base_url, section=section)
    links = extract_links(base_url, html)
    bus.publish("crawl_done", section=section, link_count=len(links),
                sample=[l["url"] for l in links[:15]])

    if RUN_STATE[section]["stop_flag"]:
        bus.publish("stopped", section=section)
        return None

    candidate = guess_finaid_link(links, host)
    if not candidate:
        bus.publish("triage_fallback", section=section,
                    message="regex triage found nothing -- asking qwen to pick a link")
        idx = ask_qwen_pick_link(links, section=section, model=model)
        candidate = links[idx] if isinstance(idx, int) and 0 <= idx < len(links) else None
    if not candidate:
        bus.publish("error", section=section, message="Could not find a financial-aid-looking page on this site.")
        return None

    bus.publish("candidate_found", section=section, url=candidate["url"], anchor_text=candidate.get("text", ""))
    return base_url, host, candidate


# ---------------------------------------------------------------------------
# FILTER PASS pipeline -- one sentence at a time, structured knowledge JSON
# ---------------------------------------------------------------------------

def run_pipeline_filter(site, model=None):
    section = "filter"
    if RUN_STATE[section]["running"]:
        bus.publish("error", section=section, message="A filter-pass run is already in progress.")
        return
    RUN_STATE[section]["running"] = True
    RUN_STATE[section]["stop_flag"] = False
    try:
        found = _find_candidate_page(section, site, model=model)
        if not found:
            return
        base_url, host, candidate = found

        if RUN_STATE[section]["stop_flag"]:
            bus.publish("stopped", section=section); return

        page_html = fetch_html(candidate["url"], section=section)
        items = walk_items(page_html)
        bus.publish("page_extracted", section=section, url=candidate["url"], item_count=len(items))

        knowledge = empty_knowledge(host)
        for i, item in enumerate(items):
            if RUN_STATE[section]["stop_flag"]:
                bus.publish("stopped", section=section); break

            messages = build_ask_messages(knowledge, item["text"], item["context"])
            full_prompt = format_prompt_for_display(messages)

            bus.publish("sentence_start", section=section, index=i, total=len(items),
                        text=item["text"], context=item["context"], prompt=full_prompt)

            delta, elapsed, raw = ask_qwen_for_sentence(messages, section=section, model=model)
            knowledge = merge_knowledge(knowledge, delta)

            bus.publish("sentence_done", section=section, index=i, total=len(items),
                        elapsed=round(elapsed, 2), delta=delta, knowledge=knowledge, raw=raw[:600],
                        prompt=full_prompt)

        bus.publish("run_done", section=section, knowledge=knowledge, stopped_early=RUN_STATE[section]["stop_flag"])
        return knowledge
    except requests.RequestException as e:
        bus.publish("error", section=section, message=f"network error: {e}")
    except Exception as e:
        bus.publish("error", section=section, message=str(e))
    finally:
        RUN_STATE[section]["running"] = False


# ---------------------------------------------------------------------------
# SNIPE pipeline -- whole paragraphs, verbatim scholarship sentences back
# ---------------------------------------------------------------------------

def run_pipeline_snipe(site, model=None):
    section = "snipe"
    if RUN_STATE[section]["running"]:
        bus.publish("error", section=section, message="A snipe run is already in progress.")
        return
    RUN_STATE[section]["running"] = True
    RUN_STATE[section]["stop_flag"] = False
    try:
        found = _find_candidate_page(section, site, model=model)
        if not found:
            return
        base_url, host, candidate = found

        if RUN_STATE[section]["stop_flag"]:
            bus.publish("stopped", section=section); return

        page_html = fetch_html(candidate["url"], section=section)
        blocks = walk_blocks(page_html)
        bus.publish("page_extracted", section=section, url=candidate["url"], item_count=len(blocks))

        all_hits = []
        for i, block in enumerate(blocks):
            if RUN_STATE[section]["stop_flag"]:
                bus.publish("stopped", section=section); break

            bus.publish("para_start", section=section, index=i, total=len(blocks),
                        text=block["text"], context=block["context"])

            spans = split_sentences_with_spans(block["text"])
            block_hit_count = 0
            block_hits = []  # hits found in THIS block, in textual order -- fed to the detail ranker below
            for j, (sentence, start, end) in enumerate(spans):
                if RUN_STATE[section]["stop_flag"]:
                    bus.publish("stopped", section=section); break

                messages = build_snipe_sentence_messages(sentence, block["context"])
                bus.publish("sentence_check_start", section=section, index=i, total=len(blocks),
                            sent_index=j, sent_total=len(spans), sentence=sentence)

                financial_aid, amount_mentioned, reason, elapsed, raw = ask_qwen_classify_sentence(
                    messages, section=section, model=model)
                category = snipe_category(financial_aid, amount_mentioned)

                if category:
                    block_hit_count += 1
                    hit = {
                        "index": i, "sentence": sentence, "category": category, "reason": reason,
                        "context": block["context"], "url": candidate["url"], "start": start, "end": end,
                    }
                    all_hits.append(hit)
                    block_hits.append(hit)

                bus.publish("sentence_check_done", section=section, index=i, total=len(blocks),
                            sent_index=j, sent_total=len(spans), sentence=sentence, start=start, end=end,
                            context=block["context"], financial_aid=financial_aid, amount_mentioned=amount_mentioned,
                            reason=reason, category=category, elapsed=round(elapsed, 2), hits_so_far=len(all_hits))

            # Once every sentence in this paragraph has been highlighted (or
            # not), compare the highlighted ones pairwise -- consecutive
            # sentences only, in the order they appear -- to grade each one
            # from 0 up to (count-1) by how much detail it carries.
            if block_hits and not RUN_STATE[section]["stop_flag"]:
                rank_block_hits_by_detail(block_hits, block["context"], section, model, i, len(blocks))

            bus.publish("para_done", section=section, index=i, total=len(blocks),
                        context=block["context"], text=block["text"],
                        hit=block_hit_count > 0, hit_count=block_hit_count, hits_so_far=len(all_hits))

        bus.publish("run_done", section=section, hits=all_hits, stopped_early=RUN_STATE[section]["stop_flag"])
        return all_hits
    except requests.RequestException as e:
        bus.publish("error", section=section, message=f"network error: {e}")
    except Exception as e:
        bus.publish("error", section=section, message=str(e))
    finally:
        RUN_STATE[section]["running"] = False


# ---------------------------------------------------------------------------
# Frontend -- single HTML page, plain CSS/JS, no build step, no CDN deps
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepship — Financial Aid Crawler</title>
<style>
  :root{
    --bg:#ffffff; --panel:#ffffff; --border:#e5e5e5; --border-strong:#d4d4d4;
    --text:#191919; --muted:#8a8a8a; --muted-strong:#5c5c5c; --accent:#191919;
    --pill-bg:#f2f2f2; --error:#b3261e; --error-bg:#fbeceb;
    --good:#1a7f37; --good-bg:#e9f7ee;
    --c-bullet:#1d4ed8; --c-heading:#b35c00; --c-paragraph:#5c5c5c;
    --c-financial_aid_overview:#2563eb; --c-scholarships:#7c3aed; --c-amounts:#059669;
    --c-deadlines:#dc2626; --c-how_to_apply:#d97706; --c-notes:#64748b;
    --snipe-accent:#7c3aed; --snipe-accent-bg:#f5f0ff;
    --c-hit-aid:#1a7f37; --c-hit-aid-bg:#dcf5e4;
    --c-hit-amount:#1d4ed8; --c-hit-amount-bg:#dfe9ff;
    --c-hit-both:#7c3aed; --c-hit-both-bg:#ede4fd;
    --detail:#b45309; --detail-bg:#fef3e2; --detail-border:#f3d19e;
    --which:#334155; --which-bg:#eef1f5;
    --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
    --mono:"SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:var(--font);}
  header{
    display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
    padding:16px 24px;border-bottom:1px solid var(--border);
  }
  .brand{display:flex;align-items:center;gap:10px;}
  .brand .mark{width:22px;height:22px;border-radius:6px;background:#191919;flex-shrink:0;}
  .brand h1{font-size:14.5px;font-weight:600;margin:0;letter-spacing:-.1px;}
  .brand .status-text{font-size:11.5px;color:var(--muted);margin-top:1px;}

  .sitebox{
    max-width:1400px;margin:20px auto 0;padding:0 24px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  }
  .sitebox label{font-size:12.5px;color:var(--muted);margin-right:2px;}
  .sitebox input{
    flex:1;min-width:220px;background:#fff;border:1px solid var(--border-strong);color:var(--text);
    border-radius:8px;padding:9px 12px;font-family:var(--font);font-size:14px;outline:none;
  }
  .sitebox input:focus{border-color:#1a1a1a;}
  .sitebox .hint{font-size:11.5px;color:var(--muted);width:100%;}
  button{
    border:1px solid var(--border-strong);background:transparent;color:var(--muted-strong);
    border-radius:8px;padding:9px 16px;cursor:pointer;font-family:var(--font);font-size:13px;
    transition:border-color .15s,color .15s,background .15s;white-space:nowrap;
  }
  button:hover:not(:disabled){color:var(--text);border-color:#1a1a1a;}
  button.primary{background:#1a1a1a;color:#fff;border-color:#1a1a1a;}
  button.primary:hover:not(:disabled){opacity:.85;}
  button.primary.snipe-btn{background:var(--snipe-accent);border-color:var(--snipe-accent);}
  button:disabled{cursor:not-allowed;opacity:.45;}

  /* ---- full-width section boxes: SNIPE and FILTER PASS each get one ---- */
  .section-box{
    max-width:1400px;margin:20px auto 0;padding:20px 22px 26px;border:1px solid var(--border);
    border-radius:16px;background:var(--panel);
  }
  .section-box:last-of-type{margin-bottom:40px;}
  .section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;
    border-bottom:1px solid var(--border);padding-bottom:14px;margin-bottom:16px;}
  .section-head .titles{display:flex;flex-direction:column;gap:3px;}
  .section-head h2.section-title{font-size:16px;font-weight:700;margin:0;letter-spacing:-.1px;}
  .section-head .section-desc{font-size:12px;color:var(--muted);max-width:640px;line-height:1.5;}
  .section-head .controls{display:flex;gap:8px;align-items:center;}
  .section-eyebrow{
    display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
    padding:2px 8px;border-radius:5px;margin-bottom:4px;width:fit-content;
  }
  .section-eyebrow.snipe{color:var(--snipe-accent);background:var(--snipe-accent-bg);}
  .section-eyebrow.filter{color:#191919;background:var(--pill-bg);}

  #snipeBanner, #filterBanner{margin-bottom:14px;}

  .twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start;}
  .col{display:flex;flex-direction:column;gap:16px;}
  @media (max-width:980px){.twocol{grid-template-columns:1fr;}}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  @media (max-width:640px){.grid{grid-template-columns:1fr;}}

  .card{border:1px solid var(--border);border-radius:12px;padding:16px 18px;background:var(--panel);}
  .card h3{font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
    color:var(--muted-strong);margin:0 0 10px;}
  .card .empty{color:var(--muted);font-size:13px;}

  .pill{
    display:inline-block;font-size:11px;padding:3px 9px;border-radius:999px;background:var(--pill-bg);
    color:var(--muted-strong);margin-bottom:8px;
  }
  .pill.bullet{background:#eef3ff;color:var(--c-bullet);}
  .pill.heading{background:#fff3e0;color:var(--c-heading);}
  .pill.paragraph{background:#f2f2f2;color:var(--c-paragraph);}

  #sentenceText, #paraText{font-size:14.5px;line-height:1.65;margin:4px 0 8px;}
  #sentenceMeta, #paraMeta{font-size:11.5px;color:var(--muted);}

  mark.snipe-hit{
    padding:1px 3px;border-radius:4px;font-weight:600;
    box-decoration-break:clone;-webkit-box-decoration-break:clone;
  }
  mark.snipe-hit.cat-aid{background:var(--c-hit-aid-bg);color:#0f5023;}
  mark.snipe-hit.cat-amount{background:var(--c-hit-amount-bg);color:#122a80;}
  mark.snipe-hit.cat-both{background:var(--c-hit-both-bg);color:#3b1a78;}

  .snipe-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-size:11.5px;color:var(--muted-strong);}
  .snipe-legend .item{display:flex;align-items:center;gap:6px;}
  .snipe-legend .swatch{width:11px;height:11px;border-radius:3px;display:inline-block;}
  .snipe-legend .swatch.cat-aid{background:var(--c-hit-aid);}
  .snipe-legend .swatch.cat-amount{background:var(--c-hit-amount);}
  .snipe-legend .swatch.cat-both{background:var(--c-hit-both);}

  mark.snipe-hit .grade-badge{
    font-size:9px;font-weight:800;vertical-align:super;margin-left:3px;opacity:.8;
  }

  .detail-top-box{
    margin-top:12px;padding:10px 12px;border-radius:8px;
    background:var(--detail-bg);border:1.4px solid var(--detail-border);
  }
  .detail-top-box .dtb-label{
    font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
    color:var(--detail);display:block;margin-bottom:4px;
  }
  .detail-top-box .dtb-text{font-size:13px;line-height:1.5;color:var(--text);font-weight:600;}
  .detail-top-box .dtb-meta{font-size:11px;color:var(--muted-strong);margin-top:4px;}

  .snipe-reason-box{
    margin-top:10px;padding:9px 12px;border-radius:8px;background:var(--pill-bg);border:1px solid var(--border);
    font-size:12px;line-height:1.55;color:var(--muted-strong);min-height:18px;
  }
  .snipe-reason-box .rb-sentence{color:var(--text);font-weight:600;display:block;margin-bottom:2px;}
  .snipe-reason-box .rb-flags{display:inline-flex;gap:8px;margin-right:8px;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em;}
  .snipe-reason-box .rb-flag-on{color:var(--c-hit-both);}
  .snipe-reason-box .rb-flag-off{color:var(--muted);}

  .model-picker{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);}
  .model-picker select{
    background:#fff;border:1px solid var(--border-strong);color:var(--text);border-radius:8px;
    padding:7px 10px;font-family:var(--font);font-size:12.5px;outline:none;max-width:220px;
  }

  progress{width:100%;height:8px;border-radius:6px;overflow:hidden;border:none;}
  progress::-webkit-progress-bar{background:var(--pill-bg);border-radius:6px;}
  progress::-webkit-progress-value{background:#1a1a1a;border-radius:6px;}
  #snipeProgressBar::-webkit-progress-value{background:var(--snipe-accent);}

  pre#knowledgeView{
    font-family:var(--mono);font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word;
    max-height:420px;overflow-y:auto;margin:0;color:var(--text);
  }
  .links-sample{font-family:var(--mono);font-size:11.5px;color:var(--muted-strong);
    max-height:120px;overflow-y:auto;line-height:1.6;}

  canvas{width:100%;height:90px;display:block;}
  .chart-label{font-size:11px;color:var(--muted);margin-bottom:4px;}

  .msg-error{color:var(--error);background:var(--error-bg);padding:10px 14px;border-radius:10px;font-size:13px;}
  .msg-good{color:var(--good);background:var(--good-bg);padding:10px 14px;border-radius:10px;font-size:13px;}

  #snipeLog, #log, #logTable{font-family:var(--mono);font-size:11.5px;line-height:1.7;color:var(--muted-strong);
    max-height:200px;overflow-y:auto;}
  #snipeLog .row, #log .row, #logTable .row{white-space:pre-wrap;word-break:break-word;}

  pre#promptView{
    font-family:var(--mono);font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word;
    max-height:220px;overflow-y:auto;margin:10px 0 0;padding:10px 12px;border-radius:8px;
    background:var(--pill-bg);color:var(--text);border:1px solid var(--border);
  }

  /* -- extracted-data boxes (filter pass) -- */
  .extracted-grid{display:flex;flex-wrap:wrap;gap:8px;max-height:230px;overflow-y:auto;align-content:flex-start;}
  .extracted-box{
    border-radius:8px;border:1.4px solid var(--border-strong);padding:8px 10px;font-size:11.5px;
    line-height:1.45;background:#fff;max-width:200px;flex:1 1 150px;
    animation:pop .18s ease-out;
  }
  @keyframes pop{ from{ transform:scale(.92); opacity:0; } to{ transform:scale(1); opacity:1; } }
  .extracted-box .exb-head{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.02em;
    margin-bottom:4px;display:flex;justify-content:space-between;gap:6px;}
  .extracted-box .exb-idx{color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0;}
  .extracted-box .exb-text{color:var(--text);word-break:break-word;}

  /* -- snipe hit list (verbatim scholarship sentences found) -- */
  .snipe-hit-list{display:flex;flex-direction:column;gap:8px;max-height:420px;overflow-y:auto;}
  .snipe-hit-card{
    border:1.4px solid var(--border-strong);border-left-width:4px;border-radius:10px;padding:10px 12px;background:#fff;
    animation:pop .18s ease-out;
  }
  .snipe-hit-card.cat-aid{border-color:var(--c-hit-aid);background:var(--c-hit-aid-bg);}
  .snipe-hit-card.cat-amount{border-color:var(--c-hit-amount);background:var(--c-hit-amount-bg);}
  .snipe-hit-card.cat-both{border-color:var(--c-hit-both);background:var(--c-hit-both-bg);}
  .snipe-hit-card .shc-head{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
    margin-bottom:5px;display:flex;justify-content:space-between;gap:8px;}
  .snipe-hit-card.cat-aid .shc-head{color:#0f5023;}
  .snipe-hit-card.cat-amount .shc-head{color:#122a80;}
  .snipe-hit-card.cat-both .shc-head{color:#3b1a78;}
  .snipe-hit-card .shc-text{font-size:13px;line-height:1.5;color:var(--text);}

  /* -- states applied to hit cards during the detail-ranking tournament -- */
  .snipe-hit-card.eliminated{opacity:.35;filter:grayscale(75%);transition:opacity .2s,filter .2s;}
  .snipe-hit-card.champion{box-shadow:0 0 0 2px var(--detail) inset;transition:box-shadow .15s;}
  .snipe-hit-card.ranked{border-left-width:5px;border-left-color:var(--detail);}
  .snipe-hit-card .shc-grade{color:var(--detail);font-weight:800;font-size:10.5px;margin-top:5px;text-transform:uppercase;letter-spacing:.03em;}

  /* -- hit list is now full width; the node graph gets its own card below -- */
  .sf-split{display:block;}

  /* ================= node-graph canvas (DaVinci-style comparison graph) ================= */
  .node-round-label{font-size:11.5px;color:var(--muted-strong);font-weight:600;line-height:1.5;margin-bottom:10px;}

  .node-canvas{
    position:relative;display:flex;align-items:stretch;gap:26px;
    padding:22px 22px 26px;min-height:340px;border-radius:14px;overflow:hidden;
    background:
      radial-gradient(circle, rgba(255,255,255,.07) 1px, transparent 1.4px) 0 0/22px 22px,
      linear-gradient(180deg,#1b1c22,#151519);
    border:1px solid #2a2b33;
  }
  .node-canvas svg.node-wires{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;}
  .node-wires path.wire{fill:none;stroke:#53555f;stroke-width:2;stroke-linecap:round;
    transition:stroke .2s,opacity .2s,d .35s ease;opacity:.6;}
  .node-wires path.wire.live{stroke:#ffb454;opacity:1;filter:drop-shadow(0 0 4px rgba(255,180,84,.65));}
  .node-wires path.wire.win{stroke:#3ddc84;opacity:1;filter:drop-shadow(0 0 4px rgba(61,220,132,.6));}
  .node-wires path.wire.loop{stroke:#b592ff;stroke-width:2.4;stroke-dasharray:6 5;opacity:0;
    filter:drop-shadow(0 0 4px rgba(181,146,255,.6));transition:opacity .25s,d .35s ease;}
  .node-wires path.wire.loop.show{opacity:.95;}
  .node-wires circle.port{fill:#4a4c56;stroke:#63656f;stroke-width:1;transition:fill .2s;}
  .node-wires circle.port.live{fill:#ffb454;}
  .node-wires circle.port.win{fill:#3ddc84;}

  .node-col{position:relative;z-index:1;display:flex;flex-direction:column;min-width:0;}
  .node-col-stack{flex:0 0 190px;}
  .node-col-which{flex:0 0 200px;justify-content:center;}
  .node-col-result{flex:0 0 190px;justify-content:center;}
  .node-col-final{flex:1 1 auto;min-width:170px;}

  .node-col-label{font-size:9.5px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;
    color:#7d7f8c;margin-bottom:10px;}

  /* the pool: a physical stack of card-nodes, front card = current pick */
  .node-stack-area{position:relative;flex:1;min-height:220px;}
  .node-stack-empty{font-size:11.5px;color:#5c5e69;padding-top:6px;line-height:1.5;}
  .stack-card{
    position:absolute;left:0;right:10px;border-radius:9px;padding:8px 10px;
    background:#26272e;border:1.4px solid #3a3c46;box-shadow:0 3px 8px rgba(0,0,0,.35);
    transition:transform .3s ease,opacity .3s ease,border-color .2s,box-shadow .2s,left .3s,right .3s;
    transform-origin:left center;
  }
  .stack-card .sc-tag{display:block;font-size:8.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
    color:#8b8d99;margin-bottom:3px;}
  .stack-card .sc-text{display:block;font-size:11.5px;line-height:1.4;color:#e7e7ec;
    display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
  .stack-card.pick{border-color:var(--detail);box-shadow:0 0 0 1.5px var(--detail) inset,0 4px 12px rgba(0,0,0,.4);}
  .stack-card.pick .sc-tag{color:var(--detail);}
  .stack-card.challenger{border-color:var(--which);}
  .stack-card.challenger .sc-tag{color:var(--which);}
  .stack-card.leaving{opacity:0!important;transform:translateX(-46px) rotate(-6deg) scale(.92)!important;}
  .stack-card.winning-out{opacity:0!important;transform:translateX(340px) scale(.9)!important;}

  /* WHICH / RESULT node boxes -- node-editor look, header bar + body */
  .ngnode{border-radius:10px;background:#26272e;border:1.4px solid #3a3c46;
    box-shadow:0 3px 10px rgba(0,0,0,.35);overflow:hidden;transition:box-shadow .2s,border-color .2s;}
  .ngnode-head{font-size:9.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
    padding:7px 10px;color:#fff;}
  .ngnode-body{padding:10px 11px 12px;font-size:12px;line-height:1.45;color:#dcdce2;min-height:44px;}
  .ng-which .ngnode-head{background:#5b4a9e;}
  .ng-which{transition:box-shadow .2s,border-color .2s;}
  .ng-which.active{border-color:#ffb454;box-shadow:0 0 0 3px rgba(255,180,84,.28),0 3px 10px rgba(0,0,0,.35);}
  .ng-which .ngnode-body{font-style:italic;color:#a9abb6;}
  .ng-result .ngnode-head{background:#1c8752;}
  .ng-result.filled{border-color:#3ddc84;}
  .ng-result.filled .ngnode-body{color:#e7e7ec;font-style:normal;}
  .ng-result .ngnode-body.empty-body{color:#5c5e69;font-style:italic;}

  .node-final{font-size:11.5px;max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;
    padding-right:2px;}
  .node-final .nf-row{display:flex;gap:7px;align-items:baseline;padding:7px 9px;border-radius:8px;
    background:#22232a;border:1px solid #34353e;animation:nfIn .25s ease;}
  @keyframes nfIn{from{opacity:0;transform:translateX(8px);}to{opacity:1;transform:translateX(0);}}
  .node-final .nf-grade{font-weight:800;color:var(--detail);flex-shrink:0;font-size:10.5px;}
  .node-final .nf-text{color:#e7e7ec;line-height:1.4;}
  .node-final .empty{color:#5c5e69;font-size:11.5px;}

  /* -- full-width sub-section: data extrapolation table (filter pass) -- */
  .extrap-wrap{overflow-x:auto;}
  table.extrap-table{width:100%;border-collapse:collapse;font-size:12.5px;}
  table.extrap-table th{
    text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
    color:var(--muted-strong);padding:8px 12px;border-bottom:1px solid var(--border-strong);
    white-space:nowrap;
  }
  table.extrap-table td{padding:12px;vertical-align:top;border-bottom:1px solid var(--border);line-height:1.5;}
  table.extrap-table tr:last-child td{border-bottom:none;}
  table.extrap-table td:nth-child(1){width:36%;color:var(--muted-strong);}
  table.extrap-table td:nth-child(2){width:24%;}
  table.extrap-table td:nth-child(3){width:40%;}
  .extrap-pill{
    display:inline-block;font-size:10.5px;font-weight:600;padding:3px 9px;border-radius:999px;
    margin:0 5px 5px 0;white-space:nowrap;
  }
  .extrap-url{display:block;margin-top:6px;font-family:var(--mono);font-size:10.5px;color:var(--muted);word-break:break-all;}
  .extrap-list{list-style:none;margin:0;padding:0;}
  .extrap-list li{display:flex;align-items:flex-start;gap:6px;margin:0 0 7px;}
  .extrap-list li:last-child{margin-bottom:0;}
  .extrap-icon{font-weight:800;font-size:14px;line-height:1.3;flex-shrink:0;}
  .extrap-item-text{color:var(--text);}

  .legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;font-size:11px;color:var(--muted-strong);}
  .legend .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle;}
  .legend .item{display:flex;align-items:center;}

  #graphCard{flex:1;}
  #graphWrap{width:100%;height:520px;}
  #nodeGraph{width:100%;height:100%;display:block;}
  #nodeGraph text{font-family:var(--font);}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="mark"></div>
    <div>
      <h1>Deepship — Financial Aid Crawler</h1>
      <div class="status-text" id="statusText">checking Ollama…</div>
    </div>
  </div>
</header>

<div class="sitebox">
  <label for="siteInput">Site to be crawled</label>
  <input id="siteInput" type="text" placeholder="e.g. princeton.edu" spellcheck="false">
  <span class="hint">Shared by both sections below — hit a section's own Run button to use it there.</span>
</div>

<!-- ======================= SNIPE SECTION ======================= -->
<div class="section-box" id="snipeSection">
  <div class="section-head">
    <div class="titles">
      <span class="section-eyebrow snipe">Snipe</span>
      <h2 class="section-title">Financial aid / scholarship / grant highlighter</h2>
      <div class="section-desc">The paragraph is shown below exactly as found on the page. In the background it's
        split into individual sentences, and each one is sent to the model on its own with two yes/no questions —
        does it talk about financial aid/scholarships/grants, and does it state an amount — plus a short reason
        (shown for context only, not used for anything). Matching sentences get highlighted in place, color-coded.</div>
    </div>
    <div class="controls">
      <div class="model-picker"><label for="snipeModelSelect">Model</label><select id="snipeModelSelect"></select></div>
      <button class="primary snipe-btn" id="snipeStartBtn">Run Snipe</button>
      <button id="snipeStopBtn" disabled>Stop</button>
    </div>
  </div>

  <div id="snipeBanner"></div>

  <div class="card" style="margin-bottom:16px;">
    <h3>Crawl</h3>
    <div id="snipeCrawlBody" class="empty">Nothing yet — enter a site above and hit Run Snipe.</div>
    <div class="links-sample" id="snipeLinksSample"></div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h3>Current paragraph <span style="color:var(--muted);font-weight:400;">(shown whole — checked one sentence at a time in the background)</span></h3>
    <div id="paraPill"></div>
    <div id="paraText" class="empty">—</div>
    <div id="paraMeta"></div>
    <progress id="snipeProgressBar" value="0" max="1"></progress>
    <div class="snipe-reason-box" id="snipeReasonBox"><span class="empty">The sentence being checked, and the model's reason, appear here as it works through the paragraph.</span></div>
    <div class="snipe-legend">
      <span class="item"><span class="swatch cat-aid"></span>talks about financial aid / a scholarship / a grant</span>
      <span class="item"><span class="swatch cat-amount"></span>states an amount</span>
      <span class="item"><span class="swatch cat-both"></span>both in one sentence</span>
    </div>
    <div class="detail-top-box" id="snipeDetailTop">
      <span class="dtb-label">Most detailed sentence in this paragraph</span>
      <span class="empty">Not enough highlighted sentences yet — appears once at least one is found and compared.</span>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h3>Sentences found <span id="snipeHitCount" style="color:var(--muted);font-weight:400;"></span></h3>
    <div class="sf-split">
      <div class="snipe-hit-list" id="snipeHitList">
        <div class="empty" id="snipeHitEmpty">Nothing found yet — a card appears here every time the model returns a matching sentence, verbatim.</div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h3>Detail-ranking node graph</h3>
    <div class="node-round-label" id="nodeRoundLabel">Once a paragraph's highlighted sentences are all found, they're stacked on the left and run through here to rank them by detail.</div>
    <div class="node-canvas" id="nodeCanvas">
      <svg class="node-wires" id="nodeWireSvg">
        <path class="wire" id="wireChamp"></path>
        <path class="wire" id="wireChall"></path>
        <path class="wire" id="wireResult"></path>
        <path class="wire loop" id="wireLoop"></path>
        <circle class="port" id="portChampOut" r="4"></circle>
        <circle class="port" id="portChallOut" r="4"></circle>
        <circle class="port" id="portWhichOut" r="4"></circle>
      </svg>
      <div class="node-col node-col-stack">
        <div class="node-col-label">Pool — stacked sentences</div>
        <div class="node-stack-area" id="nodeStackArea">
          <div class="node-stack-empty" id="nodeStackEmpty">Sentences found in this paragraph pile up here, waiting to be compared two at a time.</div>
        </div>
      </div>
      <div class="node-col node-col-which">
        <div class="node-col-label">Compare</div>
        <div class="ngnode ng-which" id="nodeWhich">
          <div class="ngnode-head">WHICH</div>
          <div class="ngnode-body">has more detail?</div>
        </div>
      </div>
      <div class="node-col node-col-result">
        <div class="node-col-label">Winner</div>
        <div class="ngnode ng-result" id="nodeResult">
          <div class="ngnode-head">RESULT</div>
          <div class="ngnode-body empty-body" id="nodeResultText">—</div>
        </div>
      </div>
      <div class="node-col node-col-final">
        <div class="node-col-label">Ranked so far</div>
        <div class="node-final" id="nodeFinal"><span class="empty">Ranked sentences will be listed here, highest detail first, as each round finishes.</span></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Event log</h3>
    <div id="snipeLog"></div>
  </div>
</div>

<!-- ======================= FILTER PASS SECTION ======================= -->
<div class="section-box" id="filterSection">
  <div class="section-head">
    <div class="titles">
      <span class="section-eyebrow filter">Filter pass</span>
      <h2 class="section-title">Structured knowledge extraction (sentence-by-sentence)</h2>
      <div class="section-desc">The original pipeline: walks the page one sentence at a time and asks the model for a
        small JSON delta each time, additively merged into a structured knowledge object.</div>
    </div>
    <div class="controls">
      <div class="model-picker"><label for="filterModelSelect">Model</label><select id="filterModelSelect"></select></div>
      <button class="primary" id="filterStartBtn">Run Filter pass</button>
      <button id="filterStopBtn" disabled>Stop</button>
    </div>
  </div>

  <div id="filterBanner"></div>

  <div class="card" style="margin-bottom:16px;">
    <h3>Crawl</h3>
    <div id="crawlBody" class="empty">Nothing yet — enter a site above and hit Run Filter pass.</div>
    <div class="links-sample" id="linksSample"></div>
  </div>

  <div class="twocol">

    <div class="col">
      <div class="grid">
        <div class="card">
          <h3>Current sentence</h3>
          <div id="contextPill"></div>
          <div id="sentenceText" class="empty">—</div>
          <div id="sentenceMeta"></div>
          <progress id="progressBar" value="0" max="1"></progress>
          <div class="chart-label" style="margin-top:10px;">Prompt sent to the model (full, this request)</div>
          <pre id="promptView" class="empty">—</pre>
        </div>
        <div class="card">
          <h3>Model round-trips</h3>
          <div class="chart-label">latency per request (s)</div>
          <canvas id="latencyChart" width="420" height="90"></canvas>
          <div class="chart-label" style="margin-top:10px;">sentences processed / total — <span style="color:#1a7f37;">green</span> = hit (useful info), <span style="color:#999;">grey</span> = miss (just another sentence)</div>
          <canvas id="progressChart" width="420" height="90"></canvas>
        </div>
      </div>

      <div class="card">
        <h3>Knowledge so far <span style="color:var(--muted);font-weight:400;">(re-sent, trimmed, on every request — grown, never lost, here)</span></h3>
        <pre id="knowledgeView" class="empty">{}</pre>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>Extracted data <span id="extractedCount" style="color:var(--muted);font-weight:400;text-transform:none;"></span></h3>
        <div id="extractedBoxes" class="extracted-grid">
          <div class="empty" id="extractedEmpty">Nothing extracted yet — a box appears here every time a sentence yields a useful fact.</div>
        </div>
      </div>

      <div class="card" id="graphCard">
        <h3>Chunk → sentence → knowledge graph</h3>
        <div id="graphWrap"><svg id="nodeGraph" viewBox="0 0 640 520"></svg></div>
        <div class="legend" id="legend"></div>
      </div>

      <div class="card">
        <h3>Event log</h3>
        <div id="log"></div>
      </div>
    </div>

  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Data extrapolation <span style="color:var(--muted);font-weight:400;text-transform:none;">— which sentence(s) produced which fact</span></h3>
    <div class="extrap-wrap">
      <table class="extrap-table">
        <thead>
          <tr><th>Source sentence(s)</th><th>Prompt · tag · URL</th><th>Extracted data</th></tr>
        </thead>
        <tbody id="extrapolationBody">
          <tr><td colspan="3" class="empty" id="extrapolationEmpty">Nothing extrapolated yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Event log</h3>
    <div id="logTable"></div>
  </div>
</div>

<script>
(function(){
  const el = (id) => document.getElementById(id);
  const siteInput = el('siteInput');
  const statusText = el('statusText');

  // ============================= SHARED ==================================
  function svgEsc(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function logRowInto(target, text){
    const row = document.createElement('div');
    row.className = 'row';
    row.textContent = text;
    target.appendChild(row);
    target.scrollTop = target.scrollHeight;
  }
  function setBannerInto(target, kind, text){
    if (!text){ target.innerHTML = ''; return; }
    target.innerHTML = '<div class="msg-' + kind + '">' + text + '</div>';
  }
  function contextKind(context){
    if (!context) return 'paragraph';
    if (context.indexOf('bullet') === 0) return 'bullet';
    if (context === 'heading') return 'heading';
    return 'paragraph';
  }
  function drawLine(canvas, values, color){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    for (let i=1;i<4;i++){
      const y = h - (h*i/4);
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
    }
    if (values.length < 2) return;
    const max = Math.max.apply(null, values.concat([0.001]));
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v,i) => {
      const x = (i/(values.length-1)) * w;
      const y = h - (v/max) * (h-10) - 5;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    const lastY = h - (values[values.length-1]/max) * (h-10) - 5;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(w-2,lastY,3,0,Math.PI*2); ctx.fill();
  }
  function drawProgressWithHits(canvas, values, hits, color, hitColor){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    for (let i=1;i<4;i++){
      const y = h - (h*i/4);
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
    }
    if (hits.length){
      const bw = w / hits.length;
      hits.forEach((hit, i) => {
        ctx.globalAlpha = hit ? 0.9 : 0.55;
        ctx.fillStyle = hit ? hitColor : '#d4d4d4';
        ctx.fillRect(i*bw, h-9, Math.max(bw-1,1), 9);
      });
      ctx.globalAlpha = 1;
    }
    if (values.length < 2) return;
    const max = Math.max.apply(null, values.concat([0.001]));
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v,i) => {
      const x = (i/(values.length-1)) * w;
      const y = h - (v/max) * (h-20) - 5;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    const lastY = h - (values[values.length-1]/max) * (h-20) - 5;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(w-2,lastY,3,0,Math.PI*2); ctx.fill();
  }

  const modelSelects = [el('snipeModelSelect'), el('filterModelSelect')].filter(Boolean);
  let modelsPopulated = false;

  function populateModelSelects(models, defaultModel){
    if (modelsPopulated) return;
    const opts = (models && models.length) ? models : [defaultModel];
    modelSelects.forEach(sel => {
      sel.innerHTML = '';
      opts.forEach(name => {
        const o = document.createElement('option');
        o.value = name; o.textContent = name;
        if (name === defaultModel) o.selected = true;
        sel.appendChild(o);
      });
    });
    modelsPopulated = true;
  }

  async function pollStatus(){
    try{
      const r = await fetch('/api/status');
      const d = await r.json();
      statusText.textContent = d.connected
        ? ('connected · model ' + d.model + (d.models && d.models.length ? ' · ' + d.models.length + ' model(s) on host' : ''))
        : ('Ollama unreachable — ' + (d.error || 'unknown error'));
      if (d.model) populateModelSelects(d.models, d.model);
    }catch(e){
      statusText.textContent = 'status check failed';
    }
  }
  pollStatus();
  setInterval(pollStatus, 8000);

  async function startSection(section, startBtn, stopBtn, resetFn, modelSelect){
    const site = siteInput.value.trim();
    if (!site){ alert('Enter a site first, e.g. princeton.edu'); return; }
    resetFn();
    startBtn.disabled = true; stopBtn.disabled = false;
    try{
      const r = await fetch('/api/' + section + '/start', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({site: site, model: modelSelect ? modelSelect.value : undefined})
      });
      if (!r.ok){
        const d = await r.json().catch(()=>({}));
        startBtn.disabled = false; stopBtn.disabled = true;
        return d.error || ('status ' + r.status);
      }
    }catch(e){
      startBtn.disabled = false; stopBtn.disabled = true;
      return e.message;
    }
    return null;
  }
  async function stopSection(section, stopBtn){
    stopBtn.disabled = true;
    await fetch('/api/' + section + '/stop', {method:'POST'});
  }

  // ============================= SNIPE ====================================
  (function(){
    const CAT_LABEL = {aid: 'financial aid / scholarship / grant', amount: 'amount stated', both: 'aid + amount'};
    const startBtn = el('snipeStartBtn'), stopBtn = el('snipeStopBtn');
    const banner = el('snipeBanner');
    const crawlBody = el('snipeCrawlBody'), linksSample = el('snipeLinksSample');
    const paraPill = el('paraPill'), paraText = el('paraText'), paraMeta = el('paraMeta');
    const progressBar = el('snipeProgressBar'), reasonBox = el('snipeReasonBox');
    const hitList = el('snipeHitList'), hitCount = el('snipeHitCount');
    let hitEmpty = el('snipeHitEmpty');
    const logEl = el('snipeLog');
    const detailTop = el('snipeDetailTop');
    const nodeRoundLabel = el('nodeRoundLabel');
    const nodeCanvas = el('nodeCanvas'), nodeWireSvg = el('nodeWireSvg');
    const nodeStackArea = el('nodeStackArea');
    let nodeStackEmpty = el('nodeStackEmpty');
    const nodeWhich = el('nodeWhich'), nodeResult = el('nodeResult'), nodeResultText = el('nodeResultText');
    const nodeFinal = el('nodeFinal');
    let nodeFinalEmpty = nodeFinal.querySelector('.empty');
    const wireChamp = el('wireChamp'), wireChall = el('wireChall'), wireResult = el('wireResult'), wireLoop = el('wireLoop');
    const portChampOut = el('portChampOut'), portChallOut = el('portChallOut'), portWhichOut = el('portWhichOut');
    let blockHitLocalCount = 0; // resets each paragraph -- position of each hit within THIS paragraph's block_hits

    let currentParaText = '';
    let currentHighlights = []; // [{start,end,category}], filled in as sentences are checked

    // -------- node graph state: the physical "pool" stack for the current round --------
    let stackPool = []; // [{idx, sentence}], front (index 0) = current pick / champion

    function reset(){
      setBannerInto(banner, null, '');
      crawlBody.textContent = 'Crawling…'; crawlBody.classList.remove('empty');
      linksSample.textContent = '';
      paraPill.innerHTML = '';
      paraText.textContent = '—'; paraText.classList.add('empty');
      paraMeta.textContent = '';
      progressBar.value = 0; progressBar.max = 1;
      reasonBox.innerHTML = '<span class="empty">The sentence being checked, and the model\\'s reason, appear here as it works through the paragraph.</span>';
      detailTop.innerHTML = '<span class="dtb-label">Most detailed sentence in this paragraph</span>' +
        '<span class="empty">Not enough highlighted sentences yet — appears once at least one is found and compared.</span>';
      currentParaText = ''; currentHighlights = [];
      blockHitLocalCount = 0;
      nodeRoundLabel.textContent = 'Once a paragraph\\'s highlighted sentences are all found, they\\'re stacked on the left and run through here to rank them by detail.';
      resetNodeGraph();
      nodeFinal.innerHTML = '<span class="empty">Ranked sentences will be listed here, highest detail first, as each round finishes.</span>';
      nodeFinalEmpty = nodeFinal.querySelector('.empty');
      logEl.innerHTML = '';
      hitList.innerHTML = '<div class="empty" id="snipeHitEmpty">Nothing found yet — a card appears here every time a sentence comes back true for aid or amount.</div>';
      hitEmpty = el('snipeHitEmpty');
      hitCount.textContent = '';
    }

    // ================= node graph rendering =================

    function resetNodeGraph(){
      stackPool = [];
      renderStack();
      setResult('—', false);
      nodeWhich.classList.remove('active');
      clearWireStates();
      requestAnimationFrame(updateWires);
    }

    function setResult(text, filled){
      nodeResultText.textContent = text;
      nodeResultText.classList.toggle('empty-body', !filled);
      nodeResult.classList.toggle('filled', filled);
    }

    function clearWireStates(){
      [wireChamp, wireChall, wireResult].forEach(w => w.classList.remove('live', 'win'));
      [portChampOut, portChallOut, portWhichOut].forEach(p => p.classList.remove('live', 'win'));
      wireLoop.classList.remove('show');
    }

    // renders stackPool as a physical stack of cards, front card on top, offset+rotated going back
    function renderStack(){
      nodeStackArea.querySelectorAll('.stack-card').forEach(c => c.remove());
      if (!stackPool.length){
        if (!nodeStackEmpty){
          nodeStackEmpty = document.createElement('div');
          nodeStackEmpty.className = 'node-stack-empty';
          nodeStackEmpty.id = 'nodeStackEmpty';
          nodeStackEmpty.textContent = 'Sentences found in this paragraph pile up here, waiting to be compared two at a time.';
          nodeStackArea.appendChild(nodeStackEmpty);
        }
        requestAnimationFrame(updateWires);
        return;
      }
      if (nodeStackEmpty){ nodeStackEmpty.remove(); nodeStackEmpty = null; }
      const shown = stackPool.slice(0, 6); // cap how many are drawn -- rest is implied depth
      shown.forEach((item, i) => {
        const card = document.createElement('div');
        card.className = 'stack-card' + (i === 0 ? ' pick' : i === 1 ? ' challenger' : '');
        card.dataset.stackIdx = String(i);
        card.style.top = (i * 14) + 'px';
        card.style.transform = 'translateX(' + (i * 4) + 'px) rotate(' + (i * -0.6) + 'deg)';
        card.style.zIndex = String(shown.length - i);
        card.style.opacity = String(Math.max(1 - i * 0.14, 0.35));
        card.innerHTML = '<span class="sc-tag">' + (i === 0 ? 'current pick' : i === 1 ? 'next sentence' : 'in queue') +
          '</span><span class="sc-text">' + svgEsc(item.sentence) + '</span>';
        nodeStackArea.appendChild(card);
      });
      requestAnimationFrame(updateWires);
    }

    // computes an anchor point (relative to nodeCanvas) on the edge of an element
    function anchor(elm, side){
      const box = elm.getBoundingClientRect(), base = nodeCanvas.getBoundingClientRect();
      const x = side === 'left' ? box.left : box.right;
      const y = box.top + box.height / 2;
      return { x: x - base.left, y: y - base.top };
    }

    function bezier(p1, p2){
      const dx = Math.max(Math.abs(p2.x - p1.x) * 0.5, 40);
      return 'M ' + p1.x + ' ' + p1.y +
        ' C ' + (p1.x + dx) + ' ' + p1.y + ', ' + (p2.x - dx) + ' ' + p2.y + ', ' + p2.x + ' ' + p2.y;
    }

    function setPort(circleEl, pt){
      circleEl.setAttribute('cx', pt.x); circleEl.setAttribute('cy', pt.y);
    }

    // redraws every wire from current DOM positions -- called after any layout change
    function updateWires(){
      const cards = nodeStackArea.querySelectorAll('.stack-card');
      const pickCard = cards[0], challCard = cards[1];
      const whichIn = anchor(nodeWhich, 'left'), whichOut = anchor(nodeWhich, 'right');
      const resultIn = anchor(nodeResult, 'left');
      const stackFallback = anchor(nodeStackArea, 'right');

      const pickPt = pickCard ? { x: anchor(pickCard, 'right').x, y: anchor(pickCard, 'right').y } : stackFallback;
      const challPt = challCard ? { x: anchor(challCard, 'right').x, y: anchor(challCard, 'right').y } : stackFallback;

      wireChamp.setAttribute('d', bezier(pickPt, { x: whichIn.x, y: whichIn.y - 9 }));
      wireChall.setAttribute('d', bezier(challPt, { x: whichIn.x, y: whichIn.y + 9 }));
      wireResult.setAttribute('d', bezier(whichOut, resultIn));
      setPort(portChampOut, pickPt);
      setPort(portChallOut, challPt);
      setPort(portWhichOut, whichOut);

      // loop-back: big arc from bottom of RESULT, dips under the canvas, back to top of the stack
      const resultBottom = { x: (anchor(nodeResult,'left').x + anchor(nodeResult,'right').x) / 2,
                              y: nodeResult.getBoundingClientRect().bottom - nodeCanvas.getBoundingClientRect().top };
      const stackTop = { x: anchor(nodeStackArea, 'left').x + 14, y: 10 };
      const dipY = Math.max(resultBottom.y, stackTop.y) + 46;
      wireLoop.setAttribute('d',
        'M ' + resultBottom.x + ' ' + resultBottom.y +
        ' C ' + resultBottom.x + ' ' + dipY + ', ' + stackTop.x + ' ' + dipY + ', ' + stackTop.x + ' ' + stackTop.y);
    }
    window.addEventListener('resize', () => requestAnimationFrame(updateWires));

    function findHitCard(paraIdx, blockIdx){
      return hitList.querySelector('.snipe-hit-card[data-para="' + paraIdx + '"][data-block-idx="' + blockIdx + '"]');
    }

    function setCardState(paraIdx, blockIdx, state, grade){
      const card = findHitCard(paraIdx, blockIdx);
      if (!card) return;
      card.classList.remove('eliminated', 'champion');
      if (state === 'eliminated') card.classList.add('eliminated');
      else if (state === 'champion') card.classList.add('champion');
      if (state === 'ranked'){
        card.classList.add('ranked');
        let badge = card.querySelector('.shc-grade');
        if (!badge){
          badge = document.createElement('div');
          badge.className = 'shc-grade';
          card.appendChild(badge);
        }
        badge.textContent = 'detail grade: ' + grade;
      }
    }

    function addHitCard(paraIdx, blockIdx, context, sentence, category){
      if (hitEmpty){ hitEmpty.remove(); hitEmpty = null; }
      const cat = CAT_LABEL[category] ? category : 'aid';
      const card = document.createElement('div');
      card.className = 'snipe-hit-card cat-' + cat;
      card.dataset.para = paraIdx;
      card.dataset.blockIdx = blockIdx;
      card.innerHTML =
        '<div class="shc-head"><span>' + svgEsc(context) + ' · ' + svgEsc(CAT_LABEL[cat]) + '</span><span>P' + (paraIdx+1) + '</span></div>' +
        '<div class="shc-text">' + svgEsc(sentence) + '</div>';
      hitList.appendChild(card);
      hitCount.textContent = '(' + hitList.querySelectorAll('.snipe-hit-card').length + ')';
      hitList.scrollTop = hitList.scrollHeight;
    }

    function renderParagraph(){
      const spans = currentHighlights.slice().sort((a,b) => a.start - b.start);
      let out = '', pos = 0;
      spans.forEach(sp => {
        if (sp.start > pos) out += svgEsc(currentParaText.slice(pos, sp.start));
        const badge = (sp.grade == null) ? '' : '<span class="grade-badge">G' + sp.grade + '</span>';
        out += '<mark class="snipe-hit cat-' + sp.category + '">' + svgEsc(currentParaText.slice(sp.start, sp.end)) + badge + '</mark>';
        pos = sp.end;
      });
      if (pos < currentParaText.length) out += svgEsc(currentParaText.slice(pos));
      paraText.innerHTML = out;
      paraText.classList.remove('empty');
    }

    function renderReason(sentence, financial_aid, amount_mentioned, reason){
      const flag = (on, label) => '<span class="' + (on ? 'rb-flag-on' : 'rb-flag-off') + '">' + label + ': ' + (on ? 'true' : 'false') + '</span>';
      reasonBox.innerHTML =
        '<span class="rb-sentence">' + svgEsc(sentence) + '</span>' +
        '<span class="rb-flags">' + flag(financial_aid, 'financial aid') + flag(amount_mentioned, 'amount') + '</span>' +
        svgEsc(reason || '(no reason given)');
    }

    startBtn.addEventListener('click', async () => {
      const err = await startSection('snipe', startBtn, stopBtn, reset, el('snipeModelSelect'));
      if (err) setBannerInto(banner, 'error', 'Could not start: ' + err);
    });
    stopBtn.addEventListener('click', () => stopSection('snipe', stopBtn));

    window.__handleSnipeEvent = function(evt){
      switch(evt.event){
        case 'crawl_start':
          logRowInto(logEl, '[' + evt.ts + '] crawling ' + evt.site);
          break;
        case 'retry':
          setBannerInto(banner, 'error', 'network hiccup on ' + evt.what + ' — retrying (attempt ' + evt.attempt + '/' + evt.max_retries + ' failed: ' + evt.message + '), waiting ' + evt.wait + 's…');
          logRowInto(logEl, '[' + evt.ts + '] retry — ' + evt.what + ' failed (' + evt.message + '), attempt ' + evt.attempt + '/' + evt.max_retries + ', waiting ' + evt.wait + 's');
          break;
        case 'crawl_done':
          crawlBody.textContent = evt.link_count + ' link(s) found on the homepage.';
          linksSample.textContent = (evt.sample || []).join('\\n');
          logRowInto(logEl, '[' + evt.ts + '] crawl done — ' + evt.link_count + ' link(s)');
          break;
        case 'triage_fallback':
          logRowInto(logEl, '[' + evt.ts + '] ' + evt.message);
          break;
        case 'candidate_found':
          crawlBody.textContent = 'Candidate financial-aid page: ' + evt.url;
          logRowInto(logEl, '[' + evt.ts + '] candidate page -> ' + evt.url);
          break;
        case 'page_extracted':
          crawlBody.textContent = 'Extracted ' + evt.item_count + ' paragraph(s) from ' + evt.url;
          progressBar.max = evt.item_count || 1;
          logRowInto(logEl, '[' + evt.ts + '] page walked -> ' + evt.item_count + ' paragraph(s)');
          break;
        case 'para_start': {
          const cls = contextKind(evt.context);
          paraPill.innerHTML = '<span class="pill ' + cls + '">' + svgEsc(evt.context) + '</span>';
          currentParaText = evt.text || '';
          currentHighlights = [];
          paraText.textContent = evt.text; paraText.classList.remove('empty');
          paraMeta.textContent = 'paragraph ' + (evt.index+1) + ' / ' + evt.total + ' — splitting into sentences…';
          blockHitLocalCount = 0;
          detailTop.innerHTML = '<span class="dtb-label">Most detailed sentence in this paragraph</span>' +
            '<span class="empty">Not enough highlighted sentences yet — appears once at least one is found and compared.</span>';
          nodeRoundLabel.textContent = 'Waiting for this paragraph\\'s sentences to finish being highlighted…';
          resetNodeGraph();
          nodeFinal.innerHTML = '<span class="empty">Ranked sentences will be listed here, highest detail first, as each round finishes.</span>';
          nodeFinalEmpty = nodeFinal.querySelector('.empty');
          break;
        }
        case 'sentence_check_start':
          paraMeta.textContent = 'paragraph ' + (evt.index+1) + ' / ' + evt.total +
            ' — checking sentence ' + (evt.sent_index+1) + ' / ' + evt.sent_total + '…';
          break;
        case 'sentence_check_done': {
          const cat = evt.category;
          if (cat) currentHighlights.push({start: evt.start, end: evt.end, category: cat});
          renderParagraph();
          renderReason(evt.sentence, evt.financial_aid, evt.amount_mentioned, evt.reason);
          paraMeta.textContent = 'paragraph ' + (evt.index+1) + ' / ' + evt.total +
            ' — sentence ' + (evt.sent_index+1) + ' / ' + evt.sent_total + ' (' + evt.elapsed + 's)' +
            (cat ? ' — matched: ' + CAT_LABEL[cat] : ' — no match');
          if (cat){
            addHitCard(evt.index, blockHitLocalCount, evt.context || '', evt.sentence, cat);
            blockHitLocalCount++;
          }
          break;
        }
        case 'detail_round_start': {
          // fresh round: un-gray everything from this paragraph that hasn't been graded yet
          hitList.querySelectorAll('.snipe-hit-card[data-para="' + evt.index + '"]').forEach(card => {
            if (!card.classList.contains('ranked')) card.classList.remove('eliminated', 'champion');
          });
          nodeRoundLabel.textContent = 'Round ' + (evt.round+1) + ' — ' + evt.pool.length + ' sentence(s) still in contention';
          // the pool physically becomes the stack -- front card is the current pick
          stackPool = (evt.pool || []).map(p => ({ idx: p.idx, sentence: p.sentence }));
          renderStack();
          if (stackPool.length) setCardState(evt.index, stackPool[0].idx, 'champion');
          setResult('—', false);
          clearWireStates();
          nodeWhich.classList.remove('active');
          break;
        }
        case 'detail_bout_start':
          // the top two cards in the stack feed into WHICH -- light up their wires
          wireChamp.classList.add('live'); portChampOut.classList.add('live');
          wireChall.classList.add('live'); portChallOut.classList.add('live');
          setResult('…thinking…', false);
          nodeWhich.classList.add('active');
          break;
        case 'detail_bout_done': {
          nodeWhich.classList.remove('active');
          wireChamp.classList.remove('live'); portChampOut.classList.remove('live');
          wireChall.classList.remove('live'); portChallOut.classList.remove('live');
          const winnerSentence = (evt.winner_idx === evt.champion_idx) ? evt.sentence_a : evt.sentence_b;
          wireResult.classList.add('win'); portWhichOut.classList.add('win');
          setResult(winnerSentence, true);
          setCardState(evt.index, evt.loser_idx, 'eliminated');
          setCardState(evt.index, evt.winner_idx, 'champion');
          // the loser physically leaves the stack; the winner slides back to the front as the new pick
          const loserCard = nodeStackArea.querySelector('.stack-card[data-stack-idx="' +
            (stackPool.findIndex(p => p.idx === evt.loser_idx)) + '"]');
          if (loserCard) loserCard.classList.add('leaving');
          const winnerEntry = stackPool.find(p => p.idx === evt.winner_idx);
          stackPool = stackPool.filter(p => p.idx !== evt.loser_idx && p.idx !== evt.winner_idx);
          if (winnerEntry) stackPool.unshift(winnerEntry);
          setTimeout(() => {
            renderStack();
            wireResult.classList.remove('win'); portWhichOut.classList.remove('win');
          }, 260);
          // show the loop-back wire briefly -- the chosen sentence loops back into WHICH to face the next one
          setTimeout(() => wireLoop.classList.add('show'), 280);
          setTimeout(() => wireLoop.classList.remove('show'), 900);
          break;
        }
        case 'detail_round_winner': {
          setCardState(evt.index, evt.winner_idx, 'ranked', evt.grade);
          if (nodeFinalEmpty){ nodeFinalEmpty.remove(); nodeFinalEmpty = null; }
          const row = document.createElement('div');
          row.className = 'nf-row';
          row.innerHTML = '<span class="nf-grade">G' + evt.grade + '</span><span class="nf-text">' + svgEsc(evt.sentence) + '</span>';
          nodeFinal.insertBefore(row, nodeFinal.firstChild);
          // the round's champion leaves the stack for good -- it's ranked now
          const wonCard = nodeStackArea.querySelector('.stack-card.pick');
          if (wonCard) wonCard.classList.add('winning-out');
          stackPool = stackPool.filter(p => p.idx !== evt.winner_idx);
          setResult('—', false);
          clearWireStates();
          setTimeout(renderStack, 260);
          if (evt.remaining_count > 0){
            nodeRoundLabel.textContent = 'Round ' + (evt.round+1) + ' winner found — ' + evt.remaining_count + ' sentence(s) left to rank';
          } else {
            nodeRoundLabel.textContent = 'Ranking complete for this paragraph.';
          }
          break;
        }
        case 'para_ranked': {
          (evt.ranked || []).forEach(r => {
            const hl = currentHighlights.find(h => h.start === r.start && h.end === r.end);
            if (hl) hl.grade = r.grade;
          });
          renderParagraph();
          if (evt.top){
            detailTop.innerHTML =
              '<span class="dtb-label">Most detailed sentence in this paragraph</span>' +
              '<span class="dtb-text">' + svgEsc(evt.top.sentence) + '</span>' +
              '<span class="dtb-meta">detail grade ' + evt.top.grade + ' of ' + (evt.count-1) +
              ' — #1 of ' + evt.count + ' highlighted sentence(s) in this paragraph</span>';
          }
          break;
        }
        case 'para_done': {
          progressBar.value = evt.index + 1;
          paraMeta.textContent = 'paragraph ' + (evt.index+1) + ' / ' + evt.total + ' done' +
            (evt.hit ? (' — ' + evt.hit_count + ' matching sentence(s) in this paragraph') : ' — nothing relevant in this one');
          break;
        }
        case 'run_done':
          setBannerInto(banner, 'good', evt.stopped_early ? 'Stopped early — ' + (evt.hits||[]).length + ' sentence(s) found so far.'
                                                            : 'Done — ' + (evt.hits||[]).length + ' sentence(s) found.');
          logRowInto(logEl, '[' + evt.ts + '] run finished' + (evt.stopped_early ? ' (stopped early)' : ''));
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
        case 'stopped':
          logRowInto(logEl, '[' + evt.ts + '] stopped by user');
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
        case 'error':
          setBannerInto(banner, 'error', evt.message);
          logRowInto(logEl, '[' + evt.ts + '] ERROR: ' + evt.message);
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
      }
    };
  })();

  // ============================= FILTER PASS ==============================
  (function(){
    const startBtn = el('filterStartBtn'), stopBtn = el('filterStopBtn');
    const banner = el('filterBanner');
    const crawlBody = el('crawlBody'), linksSample = el('linksSample');
    const contextPill = el('contextPill'), sentenceText = el('sentenceText'), sentenceMeta = el('sentenceMeta');
    const progressBar = el('progressBar'), knowledgeView = el('knowledgeView'), logEl = el('log');
    const logTableEl = el('logTable'), promptView = el('promptView');
    const latencyCanvas = el('latencyChart'), progressCanvas = el('progressChart');
    const extractedBoxes = el('extractedBoxes'), extractedCount = el('extractedCount');
    let extractedEmpty = el('extractedEmpty');
    const nodeGraph = el('nodeGraph'), legendEl = el('legend');
    const extrapolationBody = el('extrapolationBody');
    let extrapolationEmpty = el('extrapolationEmpty');

    const CATEGORY_ORDER = ['financial_aid_overview','scholarships','amounts','deadlines','how_to_apply','notes'];
    const CATEGORY_LABELS = {
      financial_aid_overview: 'Aid overview', scholarships: 'Scholarships', amounts: 'Amounts',
      deadlines: 'Deadlines', how_to_apply: 'How to apply', notes: 'Notes'
    };
    const CATEGORY_COLORS = {
      financial_aid_overview: '#2563eb', scholarships: '#7c3aed', amounts: '#059669',
      deadlines: '#dc2626', how_to_apply: '#d97706', notes: '#64748b'
    };
    const CONTEXT_COLORS = { bullet: '#1d4ed8', heading: '#b35c00', paragraph: '#5c5c5c' };
    const CONTEXT_LABELS = { bullet: 'bullet point', heading: 'heading', paragraph: 'paragraph' };

    let latencyPoints = [];
    let progressPoints = [];
    let hitMissPoints = [];
    let sentenceHistory = [];
    let extractionRecords = [];
    let candidateUrl = '';
    const MAX_VISIBLE_NODES = 9;

    function buildLegend(){
      let parts = [];
      Object.keys(CONTEXT_COLORS).forEach(k => {
        parts.push('<span class="item"><span class="dot" style="background:' + CONTEXT_COLORS[k] + '"></span>' + CONTEXT_LABELS[k] + '</span>');
      });
      CATEGORY_ORDER.forEach(k => {
        parts.push('<span class="item"><span class="dot" style="background:' + CATEGORY_COLORS[k] + '"></span>' + CATEGORY_LABELS[k] + '</span>');
      });
      legendEl.innerHTML = parts.join('');
    }
    buildLegend();

    function logRow(text){
      [logEl, logTableEl].forEach(function(target){ logRowInto(target, text); });
    }

    function pillClass(context){ return contextKind(context); }

    function redrawCharts(){
      drawLine(latencyCanvas, latencyPoints, '#1a1a1a');
      drawProgressWithHits(progressCanvas, progressPoints, hitMissPoints, '#1d4ed8', '#1a7f37');
    }

    function computeContributedKeys(delta){
      if (!delta) return [];
      return Object.keys(delta).filter(k => {
        const v = delta[k];
        if (Array.isArray(v)) return v.length > 0;
        if (typeof v === 'string') return v.trim().length > 0;
        return !!v;
      }).filter(k => CATEGORY_COLORS[k]);
    }

    function buildChunkPreview(delta){
      const keys = computeContributedKeys(delta);
      if (!keys.length) return null;
      const k = keys[0];
      let v = delta[k];
      let first = Array.isArray(v) ? v[0] : v;
      let full = typeof first === 'string' ? first : JSON.stringify(first);
      full = (full || '').trim() || '(added)';
      const text = full.length > 40 ? full.slice(0, 37) + '…' : full;
      return { key: k, text: text, full: full };
    }

    function addExtractedBox(index, chunk){
      if (extractedEmpty){ extractedEmpty.remove(); extractedEmpty = null; }
      const color = CATEGORY_COLORS[chunk.key] || '#191919';
      const box = document.createElement('div');
      box.className = 'extracted-box';
      box.style.borderColor = color;
      box.title = (CATEGORY_LABELS[chunk.key] || chunk.key) + ': ' + chunk.full;
      box.innerHTML =
        '<div class="exb-head" style="color:' + color + '">' + svgEsc(CATEGORY_LABELS[chunk.key] || chunk.key) +
        '<span class="exb-idx">S' + (index+1) + '</span></div>' +
        '<div class="exb-text">' + svgEsc(chunk.text) + '</div>';
      extractedBoxes.appendChild(box);
      extractedCount.textContent = '(' + extractedBoxes.children.length + ')';
      extractedBoxes.scrollTop = extractedBoxes.scrollHeight;
    }

    function updateExtractionRecords(entry, delta){
      const keys = computeContributedKeys(delta);
      if (!keys.length) return;
      const primaryKey = keys[0];
      const items = [];
      keys.forEach(k => {
        const v = delta[k];
        const arr = Array.isArray(v) ? v : [v];
        arr.forEach(item => {
          const text = typeof item === 'string' ? item : JSON.stringify(item);
          if (text && text.trim()) items.push({ key: k, text: text.trim() });
        });
      });

      const last = extractionRecords[extractionRecords.length - 1];
      if (last && last.key === primaryKey && entry.index === last.lastIndex + 1){
        last.sentences.push(entry.text || '(sentence ' + (entry.index+1) + ')');
        last.items = last.items.concat(items);
        last.lastIndex = entry.index;
      } else {
        extractionRecords.push({
          key: primaryKey,
          context: entry.context,
          url: candidateUrl,
          sentences: [entry.text || '(sentence ' + (entry.index+1) + ')'],
          items: items,
          firstIndex: entry.index,
          lastIndex: entry.index,
        });
      }
      renderExtrapolationTable();
    }

    function renderExtrapolationTable(){
      if (!extractionRecords.length){
        extrapolationBody.innerHTML = '<tr><td colspan="3" class="empty" id="extrapolationEmpty">Nothing extrapolated yet.</td></tr>';
        extrapolationEmpty = el('extrapolationEmpty');
        return;
      }
      if (extrapolationEmpty){ extrapolationEmpty = null; }

      const rows = extractionRecords.map(rec => {
        const catColor = CATEGORY_COLORS[rec.key] || '#191919';
        const ctxKind = contextKind(rec.context);
        const ctxColor = CONTEXT_COLORS[ctxKind] || '#5c5c5c';
        const label = rec.sentences.length > 1
          ? 'S' + (rec.firstIndex+1) + '–S' + (rec.lastIndex+1) + ' (' + rec.sentences.length + ' sentences)'
          : 'S' + (rec.firstIndex+1);

        const col1 = '<div style="font-size:10.5px;color:var(--muted);margin-bottom:4px;">' + svgEsc(label) + '</div>' +
          svgEsc(rec.sentences.join(' '));

        const col2 =
          '<span class="extrap-pill" style="background:' + ctxColor + '19;color:' + ctxColor + ';border:1px solid ' + ctxColor + '55;">prompt: ' + svgEsc(CONTEXT_LABELS[ctxKind] || ctxKind) + ' sentence</span>' +
          '<span class="extrap-pill" style="background:' + catColor + '19;color:' + catColor + ';border:1px solid ' + catColor + '55;">tag: ' + svgEsc(CATEGORY_LABELS[rec.key] || rec.key) + '</span>' +
          (rec.url ? '<span class="extrap-url">' + svgEsc(rec.url) + '</span>' : '');

        const uniqueItems = [];
        const seen = new Set();
        rec.items.forEach(it => { if (!seen.has(it.text)){ seen.add(it.text); uniqueItems.push(it); } });
        const col3 = '<ul class="extrap-list">' + uniqueItems.map(it =>
          '<li><span class="extrap-icon" style="color:' + (CATEGORY_COLORS[it.key] || '#191919') + ';">&rsaquo;</span><span class="extrap-item-text">' + svgEsc(it.text) + '</span></li>'
        ).join('') + '</ul>';

        return '<tr><td>' + col1 + '</td><td>' + col2 + '</td><td>' + col3 + '</td></tr>';
      });

      extrapolationBody.innerHTML = rows.join('');
    }

    function renderGraph(){
      const W = 640, H = 520;
      const catX = W - 128;
      const catGap = H / (CATEGORY_ORDER.length + 1);
      const catPos = {};
      CATEGORY_ORDER.forEach((cat, i) => { catPos[cat] = { x: catX, y: catGap * (i+1) }; });

      const visible = sentenceHistory.slice(-MAX_VISIBLE_NODES);
      const sentX = 156;
      const sentGap = H / (MAX_VISIBLE_NODES + 1);
      const chunkX = 16;
      const chunkW = 116;

      let parts = [];

      visible.forEach((s, i) => {
        if (!s.chunkPreview) return;
        const sy = sentGap * (i+1);
        const x1 = chunkX + chunkW, y1 = sy, x2 = sentX, y2 = sy;
        const mx = (x1+x2)/2;
        const d = 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 + ', ' + x2 + ' ' + y2;
        const color = CATEGORY_COLORS[s.chunkPreview.key] || '#999';
        parts.push('<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.4" opacity="0.45"/>');
      });

      visible.forEach((s, i) => {
        const sy = sentGap * (i+1);
        (s.contributedKeys || []).forEach(key => {
          const cp = catPos[key];
          if (!cp) return;
          const x1 = sentX + 92, y1 = sy, x2 = cp.x, y2 = cp.y;
          const mx = (x1+x2)/2;
          const d = 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 + ', ' + x2 + ' ' + y2;
          parts.push('<path d="' + d + '" fill="none" stroke="' + CATEGORY_COLORS[key] + '" stroke-width="1.6" opacity="0.55"/>');
        });
      });

      CATEGORY_ORDER.forEach(cat => {
        const p = catPos[cat], color = CATEGORY_COLORS[cat];
        parts.push(
          '<rect x="' + p.x + '" y="' + (p.y-14) + '" width="118" height="28" rx="8" fill="#ffffff" stroke="' + color + '" stroke-width="1.6"/>' +
          '<circle cx="' + p.x + '" cy="' + p.y + '" r="4" fill="' + color + '"/>' +
          '<text x="' + (p.x+12) + '" y="' + (p.y+4) + '" font-size="11" fill="' + color + '">' + svgEsc(CATEGORY_LABELS[cat]) + '</text>'
        );
      });

      visible.forEach((s, i) => {
        const sy = sentGap * (i+1);
        const kind = contextKind(s.context);
        const color = CONTEXT_COLORS[kind];
        const hasDelta = (s.contributedKeys || []).length > 0;
        parts.push(
          '<circle cx="' + (sentX+92) + '" cy="' + sy + '" r="4" fill="' + color + '"/>' +
          '<rect x="' + sentX + '" y="' + (sy-14) + '" width="92" height="28" rx="8" fill="#ffffff" stroke="' + color +
            '" stroke-width="' + (hasDelta ? 1.8 : 1) + '" opacity="' + (hasDelta ? 1 : 0.55) + '"/>' +
          '<text x="' + (sentX+8) + '" y="' + (sy+4) + '" font-size="10" fill="' + color + '">S' + (s.index+1) + ' · ' + svgEsc(kind) + '</text>'
        );
      });

      visible.forEach((s, i) => {
        if (!s.chunkPreview) return;
        const sy = sentGap * (i+1);
        const color = CATEGORY_COLORS[s.chunkPreview.key] || '#999';
        const clipId = 'clip-chunk-' + s.index;
        const tooltip = (CATEGORY_LABELS[s.chunkPreview.key] || '') + ': ' + s.chunkPreview.full;
        parts.push(
          '<clipPath id="' + clipId + '"><rect x="' + (chunkX+6) + '" y="' + (sy-13) + '" width="' + (chunkW-12) + '" height="26"/></clipPath>' +
          '<g>' +
            '<title>' + svgEsc(tooltip) + '</title>' +
            '<rect x="' + chunkX + '" y="' + (sy-15) + '" width="' + chunkW + '" height="30" rx="8" fill="#fbfbfb" stroke="' + color + '" stroke-width="1.4"/>' +
            '<text clip-path="url(#' + clipId + ')" x="' + (chunkX+8) + '" y="' + (sy-2) + '" font-size="8.5" fill="' + color + '" font-weight="700">' + svgEsc(CATEGORY_LABELS[s.chunkPreview.key] || '') + '</text>' +
            '<text clip-path="url(#' + clipId + ')" x="' + (chunkX+8) + '" y="' + (sy+9) + '" font-size="8.5" fill="#555">' + svgEsc(s.chunkPreview.text) + '</text>' +
          '</g>'
        );
      });

      nodeGraph.innerHTML = parts.join('');
    }

    function renderKnowledgeHTML(knowledge){
      let text = JSON.stringify(knowledge, null, 2);
      text = svgEsc(text);
      CATEGORY_ORDER.forEach(key => {
        const color = CATEGORY_COLORS[key];
        text = text.replace(new RegExp('"' + key + '"', 'g'),
          '<span style="color:' + color + ';font-weight:600;">"' + key + '"</span>');
      });
      return text;
    }

    function reset(){
      setBannerInto(banner, null, '');
      crawlBody.textContent = 'Crawling…'; crawlBody.classList.remove('empty');
      linksSample.textContent = '';
      contextPill.innerHTML = '';
      sentenceText.textContent = '—'; sentenceText.classList.add('empty');
      sentenceMeta.textContent = '';
      progressBar.value = 0; progressBar.max = 1;
      promptView.textContent = '—'; promptView.classList.add('empty');
      knowledgeView.textContent = '{}';
      logEl.innerHTML = ''; logTableEl.innerHTML = '';
      latencyPoints = []; progressPoints = []; hitMissPoints = []; sentenceHistory = [];
      extractionRecords = []; candidateUrl = '';
      extractedBoxes.innerHTML = '<div class="empty" id="extractedEmpty">Nothing extracted yet — a box appears here every time a sentence yields a useful fact.</div>';
      extractedEmpty = el('extractedEmpty');
      extractedCount.textContent = '';
      extrapolationBody.innerHTML = '<tr><td colspan="3" class="empty" id="extrapolationEmpty">Nothing extrapolated yet.</td></tr>';
      extrapolationEmpty = el('extrapolationEmpty');
      redrawCharts(); renderGraph();
    }

    startBtn.addEventListener('click', async () => {
      const err = await startSection('filter', startBtn, stopBtn, reset, el('filterModelSelect'));
      if (err) setBannerInto(banner, 'error', 'Could not start: ' + err);
    });
    stopBtn.addEventListener('click', () => stopSection('filter', stopBtn));

    window.__handleFilterEvent = function(evt){
      switch(evt.event){
        case 'crawl_start':
          logRow('[' + evt.ts + '] crawling ' + evt.site);
          break;
        case 'retry':
          setBannerInto(banner, 'error', 'network hiccup on ' + evt.what + ' — retrying (attempt ' + evt.attempt + '/' + evt.max_retries + ' failed: ' + evt.message + '), waiting ' + evt.wait + 's…');
          logRow('[' + evt.ts + '] retry — ' + evt.what + ' failed (' + evt.message + '), attempt ' + evt.attempt + '/' + evt.max_retries + ', waiting ' + evt.wait + 's');
          break;
        case 'crawl_done':
          crawlBody.textContent = evt.link_count + ' link(s) found on the homepage.';
          linksSample.textContent = (evt.sample || []).join('\\n');
          logRow('[' + evt.ts + '] crawl done — ' + evt.link_count + ' link(s)');
          break;
        case 'triage_fallback':
          logRow('[' + evt.ts + '] ' + evt.message);
          break;
        case 'candidate_found':
          crawlBody.textContent = 'Candidate financial-aid page: ' + evt.url;
          logRow('[' + evt.ts + '] candidate page -> ' + evt.url);
          candidateUrl = evt.url;
          break;
        case 'page_extracted':
          crawlBody.textContent = 'Extracted ' + evt.item_count + ' sentence(s) from ' + evt.url;
          progressBar.max = evt.item_count || 1;
          logRow('[' + evt.ts + '] page walked -> ' + evt.item_count + ' sentence(s)');
          break;
        case 'sentence_start': {
          const cls = pillClass(evt.context);
          contextPill.innerHTML = '<span class="pill ' + cls + '">' + svgEsc(evt.context) + '</span>';
          sentenceText.textContent = evt.text;
          sentenceText.classList.remove('empty');
          sentenceMeta.textContent = 'sentence ' + (evt.index+1) + ' / ' + evt.total + ' — asking qwen…';
          if (evt.prompt){
            promptView.textContent = evt.prompt;
            promptView.classList.remove('empty');
          }
          sentenceHistory.push({index: evt.index, context: evt.context, contributedKeys: [], text: evt.text, prompt: evt.prompt});
          renderGraph();
          break;
        }
        case 'sentence_done': {
          sentenceMeta.textContent = 'sentence ' + (evt.index+1) + ' / ' + evt.total +
            ' — ' + evt.elapsed + 's round-trip';
          progressBar.value = evt.index + 1;
          knowledgeView.innerHTML = renderKnowledgeHTML(evt.knowledge);
          knowledgeView.classList.remove('empty');
          latencyPoints.push(evt.elapsed);
          progressPoints.push((evt.index+1) / evt.total);

          const keys = computeContributedKeys(evt.delta);
          let entry = sentenceHistory.find(s => s.index === evt.index);
          if (!entry){ entry = {index: evt.index, context: evt.context}; sentenceHistory.push(entry); }
          entry.contributedKeys = keys;
          entry.chunkPreview = buildChunkPreview(evt.delta);

          hitMissPoints.push(keys.length > 0);
          redrawCharts();
          renderGraph();

          if (entry.chunkPreview){
            addExtractedBox(evt.index, entry.chunkPreview);
            updateExtractionRecords(entry, evt.delta);
          }
          break;
        }
        case 'run_done':
          setBannerInto(banner, 'good', evt.stopped_early ? 'Stopped early — knowledge collected so far is shown above.'
                                               : 'Done — knowledge JSON above is the full result.');
          logRow('[' + evt.ts + '] run finished' + (evt.stopped_early ? ' (stopped early)' : ''));
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
        case 'stopped':
          logRow('[' + evt.ts + '] stopped by user');
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
        case 'error':
          setBannerInto(banner, 'error', evt.message);
          logRow('[' + evt.ts + '] ERROR: ' + evt.message);
          startBtn.disabled = false; stopBtn.disabled = true;
          break;
      }
    };

    redrawCharts();
    renderGraph();
  })();

  // ============================= SHARED SSE DISPATCH =======================
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    let evt;
    try{ evt = JSON.parse(e.data); } catch(err){ return; }
    if (evt.section === 'snipe' && window.__handleSnipeEvent) window.__handleSnipeEvent(evt);
    else if (evt.section === 'filter' && window.__handleFilterEvent) window.__handleFilterEvent(evt);
  };
  es.onerror = () => { /* browser auto-reconnects SSE */ };
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "Deepship/3.0"

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; the browser shows everything

    def _send(self, code, body, content_type="text/plain; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._status()
        elif self.path == "/api/events":
            self._sse()
        else:
            self._send(404, "not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = {}

        if self.path == "/api/snipe/start":
            self._start("snipe", run_pipeline_snipe, body)
        elif self.path == "/api/snipe/stop":
            self._stop("snipe")
        elif self.path == "/api/filter/start":
            self._start("filter", run_pipeline_filter, body)
        elif self.path == "/api/filter/stop":
            self._stop("filter")
        else:
            self._send(404, "not found")

    def _start(self, section, pipeline_fn, body):
        site = (body.get("site") or "").strip()
        model = (body.get("model") or "").strip() or None
        if RUN_STATE[section]["running"]:
            self._send(409, json.dumps({"error": f"a {section} run is already in progress"}), "application/json")
            return
        threading.Thread(target=pipeline_fn, args=(site, model), daemon=True).start()
        self._send(200, json.dumps({"ok": True}), "application/json")

    def _stop(self, section):
        RUN_STATE[section]["stop_flag"] = True
        self._send(200, json.dumps({"ok": True}), "application/json")

    def _status(self):
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=5)
            r.raise_for_status()
            names = [m.get("name") for m in r.json().get("models", [])]
            self._send(200, json.dumps({"connected": True, "models": names, "model": OLLAMA_MODEL}),
                       "application/json")
        except Exception as e:
            self._send(200, json.dumps({"connected": False, "error": str(e), "model": OLLAMA_MODEL}),
                       "application/json")

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = bus.subscribe()
        try:
            while True:
                try:
                    payload = q.get(timeout=15)
                    self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            bus.unsubscribe(q)


def run_server(host="127.0.0.1", port=8765, open_browser=True):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Deepship running at {url}")
    print(f"Ollama host: {OLLAMA_BASE_URL}  model: {OLLAMA_MODEL}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


# ---------------------------------------------------------------------------
# Headless CLI (writes a JSON file, no browser) -- runs the filter pass
# ---------------------------------------------------------------------------

def _run_cli(site, output_path, mode="filter"):
    section = "snipe" if mode == "snipe" else "filter"
    pipeline_fn = run_pipeline_snipe if section == "snipe" else run_pipeline_filter
    done = threading.Event()
    result = {}

    def listener():
        q = bus.subscribe()
        while True:
            payload = q.get()
            evt = json.loads(payload)
            if evt.get("section") != section:
                continue
            name = evt.get("event")
            if name in ("sentence_done", "para_done"):
                extra = evt.get("delta") if name == "sentence_done" else evt.get("sentences")
                idx, total, elapsed = evt.get("index"), evt.get("total"), evt.get("elapsed")
                print(f"  [{idx+1}/{total}] ({elapsed}s) {extra}")
            elif name in ("crawl_start", "crawl_done", "triage_fallback",
                          "candidate_found", "page_extracted"):
                extra = {k: v for k, v in evt.items() if k not in ("event", "ts", "section")}
                print(f"-- {name}: {extra}")
            elif name == "run_done":
                result["data"] = evt.get("knowledge") if section == "filter" else evt.get("hits")
                done.set()
                return
            elif name in ("stopped", "error"):
                if name == "error":
                    print(f"! error: {evt.get('message')}", file=sys.stderr)
                done.set()
                return

    threading.Thread(target=listener, daemon=True).start()
    pipeline_fn(site)
    done.wait()

    if result.get("data") is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["data"], f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_path}")
    else:
        print("\nNo result produced.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crawl ONE college site, find its financial-aid/scholarships page, and "
                     "either (a) snipe verbatim scholarship sentences out of whole paragraphs, or "
                     "(b) run the full sentence-by-sentence structured filter pass, via a local "
                     "qwen2.5 Ollama server. Run with no arguments to launch the live web GUI "
                     "instead, where each of the two is its own section with its own Run button."
    )
    parser.add_argument("site", nargs="?", help="College domain, e.g. princeton.edu (CLI mode)")
    parser.add_argument("--output", default="deepship_result.json", help="Where to write JSON (CLI mode)")
    parser.add_argument("--mode", choices=["filter", "snipe"], default="filter",
                        help="CLI mode: 'filter' (default, structured knowledge JSON) or "
                             "'snipe' (verbatim scholarship sentences)")
    parser.add_argument("--port", type=int, default=8765, help="Port for the web GUI (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    args = parser.parse_args()

    if args.site:
        _run_cli(args.site, args.output, mode=args.mode)
    else:
        run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()