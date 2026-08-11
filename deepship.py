#!/usr/bin/env python3
"""
deepship.py  (v2 — single-site, sentence-by-sentence, browser GUI)
--------------------------------------------------------------------
Revamped pipeline for pulling financial-aid / scholarship facts out of ONE
college site at a time, talking to a local Ollama server running
qwen2.5:1.5b over Tailscale. GUI is now plain HTML/CSS/JS served by a tiny
built-in Python web server (no Tkinter, no external JS/CSS frameworks) with
live updates pushed over Server-Sent Events (SSE).

Pipeline, per site:

  1. CRAWL     — fetch the homepage, pull every <a href> out of it with
                 BeautifulSoup.
  2. TRIAGE    — regex-score same-domain links for finaid/scholarship-ish
                 words first (cheap, no model call). Only if nothing scores
                 falls back to ONE minimal call to qwen with a trimmed
                 {index, anchor-text} list, asking for the best index.
  3. FETCH     — fetch the chosen page.
  4. WALK      — walk the page's DOM in document order with BeautifulSoup
                 and turn it into a flat list of *single sentences*, each
                 carrying a structural context tag:
                   "heading"
                   "paragraph"
                   "bullet-point (start of a list)"
                   "bullet-point (#3 in the same list as the previous
                    bullet(s))"
                 so a run of consecutive bullets is explicitly told to the
                 model as such, instead of the model having to guess from
                 stripped text.
  5. LEARN     — one sentence at a time (never more), send the model a
                 MINIMAL request:
                     { "known": <trimmed knowledge so far>,
                       "context": <tag from step 4>,
                       "sentence": <this one sentence> }
                 and ask it to reply with ONLY the JSON delta it can add.
                 The "known" we hand the model is a trimmed window (last
                 few items per list) so requests stay small and fast even
                 on a slow local TPS -- but the CANONICAL knowledge JSON
                 kept by the Python side never shrinks: every reply is
                 additively merged into it (new items appended, dupes
                 dropped), so nothing already learned gets lost just
                 because the model didn't see it again on this request.

Every step streams live to the browser: crawl progress, the current
sentence + its context tag, each model round-trip's latency, and the
knowledge JSON as it grows -- plus two small live charts (latency per
request, and sentences processed so far / total).

Requirements:
    pip install requests beautifulsoup4

Config (env vars, or a .env file sitting next to this script):
    OLLAMA_BASE_URL=https://offr.tail05ae98.ts.net   (your Tailscale Ollama host)
    OLLAMA_MODEL=qwen2.5:1.5b

Usage:
    python3 deepship.py                    # launches the web GUI (opens browser)
    python3 deepship.py --port 9000        # same, on a different port
    python3 deepship.py princeton.edu      # headless CLI, writes a JSON file
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

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (deepship/2.0 crawler)"}
REQUEST_TIMEOUT = 20        # page fetches
MODEL_TIMEOUT = 120         # local model round-trips can be slow -- be patient
MAX_RETRIES = 3             # network/model calls get this many attempts total
RETRY_BACKOFF = 2.0         # seconds, doubles each retry (2, 4, 8...)


# ---------------------------------------------------------------------------
# Knowledge JSON -- the thing that gets filled up and re-ingested every call
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
# Ollama calls -- always ONE sentence, always minimal JSON
# ---------------------------------------------------------------------------

def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _with_retries(fn, what, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
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
                bus.publish("retry", what=what, attempt=attempt, max_retries=max_retries,
                            wait=round(wait, 1), message=str(e))
            except NameError:
                pass  # bus not defined yet (shouldn't happen once module is loaded)
            time.sleep(wait)
    raise last_exc


def _call_ollama(messages, timeout=MODEL_TIMEOUT):
    payload = {
        "model": OLLAMA_MODEL,
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

    return _with_retries(attempt, "model round-trip")


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


def ask_qwen_for_sentence(knowledge, sentence, context_tag):
    trimmed_known = trim_knowledge_for_prompt(knowledge)
    user_payload = {"known": trimmed_known, "context": context_tag, "sentence": sentence}
    messages = [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    raw, elapsed = _call_ollama(messages)
    try:
        delta = json.loads(raw)
        if not isinstance(delta, dict):
            delta = {}
    except Exception:
        delta = {}
    return delta, elapsed, raw


def ask_qwen_pick_link(links):
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
        raw, _ = _call_ollama(messages, timeout=60)
        idx = json.loads(raw).get("i")
        return idx if isinstance(idx, int) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Crawling / extraction
# ---------------------------------------------------------------------------

def fetch_html(url):
    def attempt():
        r = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text

    return _with_retries(attempt, f"fetch {url}")


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


def walk_items(html):
    """Walk the page in document order, flattening it into single sentences,
    each tagged with structural context -- headings, plain paragraphs, and
    bullet points explicitly numbered within their own <ul>/<ol> run so
    consecutive bullets are called out as such."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    items = []
    bullet_pos = {}  # id(parent list) -> how many bullets seen so far in it

    def emit(block_text, context):
        for s in split_sentences(block_text):
            items.append({"text": s, "context": context})

    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name.startswith("h"):
            emit(txt, "heading")
        elif el.name == "li":
            parent_list = el.find_parent(["ul", "ol"])
            key = id(parent_list) if parent_list else id(el)
            bullet_pos[key] = bullet_pos.get(key, 0) + 1
            pos = bullet_pos[key]
            tag = ("bullet-point (start of a list)" if pos == 1 else
                   f"bullet-point (#{pos} in the same list as the previous bullet(s))")
            emit(txt, tag)
        else:
            emit(txt, "paragraph")
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
RUN_STATE = {"running": False, "stop_flag": False}


# ---------------------------------------------------------------------------
# The pipeline itself -- single site, one sentence at a time
# ---------------------------------------------------------------------------

def run_pipeline(site):
    if RUN_STATE["running"]:
        bus.publish("error", message="A run is already in progress.")
        return
    RUN_STATE["running"] = True
    RUN_STATE["stop_flag"] = False
    try:
        site = (site or "").strip()
        if not site:
            bus.publish("error", message="No site given.")
            return
        base_url = site if site.startswith("http") else f"https://{site}"
        host = urlparse(base_url).netloc or site

        bus.publish("crawl_start", site=site)
        html = fetch_html(base_url)
        links = extract_links(base_url, html)
        bus.publish("crawl_done", link_count=len(links),
                    sample=[l["url"] for l in links[:15]])

        if RUN_STATE["stop_flag"]:
            bus.publish("stopped"); return

        candidate = guess_finaid_link(links, host)
        if not candidate:
            bus.publish("triage_fallback", message="regex triage found nothing -- asking qwen to pick a link")
            idx = ask_qwen_pick_link(links)
            candidate = links[idx] if isinstance(idx, int) and 0 <= idx < len(links) else None
        if not candidate:
            bus.publish("error", message="Could not find a financial-aid-looking page on this site.")
            return

        bus.publish("candidate_found", url=candidate["url"], anchor_text=candidate.get("text", ""))

        if RUN_STATE["stop_flag"]:
            bus.publish("stopped"); return

        page_html = fetch_html(candidate["url"])
        items = walk_items(page_html)
        bus.publish("page_extracted", url=candidate["url"], item_count=len(items))

        knowledge = empty_knowledge(host)
        for i, item in enumerate(items):
            if RUN_STATE["stop_flag"]:
                bus.publish("stopped"); break

            bus.publish("sentence_start", index=i, total=len(items),
                        text=item["text"], context=item["context"])

            delta, elapsed, raw = ask_qwen_for_sentence(knowledge, item["text"], item["context"])
            knowledge = merge_knowledge(knowledge, delta)

            bus.publish("sentence_done", index=i, total=len(items),
                        elapsed=round(elapsed, 2), delta=delta, knowledge=knowledge, raw=raw[:600])

        bus.publish("run_done", knowledge=knowledge, stopped_early=RUN_STATE["stop_flag"])
        return knowledge
    except requests.RequestException as e:
        bus.publish("error", message=f"network error: {e}")
    except Exception as e:
        bus.publish("error", message=str(e))
    finally:
        RUN_STATE["running"] = False


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
  button{
    border:1px solid var(--border-strong);background:transparent;color:var(--muted-strong);
    border-radius:8px;padding:9px 16px;cursor:pointer;font-family:var(--font);font-size:13px;
    transition:border-color .15s,color .15s,background .15s;
  }
  button:hover:not(:disabled){color:var(--text);border-color:#1a1a1a;}
  button.primary{background:#1a1a1a;color:#fff;border-color:#1a1a1a;}
  button.primary:hover:not(:disabled){opacity:.85;}
  button:disabled{cursor:not-allowed;opacity:.45;}

  #banner{max-width:1400px;margin:12px auto 0;padding:0 24px;}

  .twocol{
    max-width:1400px;margin:0 auto;padding:16px 24px 60px;
    display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start;
  }
  .col{display:flex;flex-direction:column;gap:16px;}
  @media (max-width:980px){.twocol{grid-template-columns:1fr;}}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  @media (max-width:640px){.grid{grid-template-columns:1fr;}}

  .card{border:1px solid var(--border);border-radius:12px;padding:16px 18px;background:var(--panel);}
  .card h2{font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
    color:var(--muted-strong);margin:0 0 10px;}
  .card .empty{color:var(--muted);font-size:13px;}

  .pill{
    display:inline-block;font-size:11px;padding:3px 9px;border-radius:999px;background:var(--pill-bg);
    color:var(--muted-strong);margin-bottom:8px;
  }
  .pill.bullet{background:#eef3ff;color:var(--c-bullet);}
  .pill.heading{background:#fff3e0;color:var(--c-heading);}
  .pill.paragraph{background:#f2f2f2;color:var(--c-paragraph);}

  #sentenceText{font-size:14.5px;line-height:1.55;margin:4px 0 8px;}
  #sentenceMeta{font-size:11.5px;color:var(--muted);}

  progress{width:100%;height:8px;border-radius:6px;overflow:hidden;border:none;}
  progress::-webkit-progress-bar{background:var(--pill-bg);border-radius:6px;}
  progress::-webkit-progress-value{background:#1a1a1a;border-radius:6px;}

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

  #log{font-family:var(--mono);font-size:11.5px;line-height:1.7;color:var(--muted-strong);
    max-height:200px;overflow-y:auto;}
  #log .row{white-space:pre-wrap;word-break:break-word;}

  /* -- right column: extracted-data boxes + node graph -- */
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
  <button class="primary" id="startBtn">Start</button>
  <button id="stopBtn" disabled>Stop</button>
</div>

<div id="banner"></div>

<div class="twocol">

  <!-- LEFT: everything the dashboard already showed -->
  <div class="col">
    <div class="card">
      <h2>Crawl</h2>
      <div id="crawlBody" class="empty">Nothing yet — enter a site and hit Start.</div>
      <div class="links-sample" id="linksSample"></div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Current sentence</h2>
        <div id="contextPill"></div>
        <div id="sentenceText" class="empty">—</div>
        <div id="sentenceMeta"></div>
        <progress id="progressBar" value="0" max="1"></progress>
      </div>
      <div class="card">
        <h2>Model round-trips</h2>
        <div class="chart-label">latency per request (s)</div>
        <canvas id="latencyChart" width="420" height="90"></canvas>
        <div class="chart-label" style="margin-top:10px;">sentences processed / total — <span style="color:#1a7f37;">green</span> = hit (useful info), <span style="color:#999;">grey</span> = miss (just another sentence)</div>
        <canvas id="progressChart" width="420" height="90"></canvas>
      </div>
    </div>

    <div class="card">
      <h2>Knowledge so far <span style="color:var(--muted);font-weight:400;">(re-sent, trimmed, on every request — grown, never lost, here)</span></h2>
      <pre id="knowledgeView" class="empty">{}</pre>
    </div>

    <div class="card">
      <h2>Event log</h2>
      <div id="log"></div>
    </div>
  </div>

  <!-- RIGHT: extracted-data boxes, and a node graph of chunk -> sentence -> fact -->
  <div class="col">
    <div class="card">
      <h2>Extracted data <span id="extractedCount" style="color:var(--muted);font-weight:400;text-transform:none;"></span></h2>
      <div id="extractedBoxes" class="extracted-grid">
        <div class="empty" id="extractedEmpty">Nothing extracted yet — a box appears here every time a sentence yields a useful fact.</div>
      </div>
    </div>

    <div class="card" id="graphCard">
      <h2>Chunk → sentence → knowledge graph</h2>
      <div id="graphWrap"><svg id="nodeGraph" viewBox="0 0 640 520"></svg></div>
      <div class="legend" id="legend"></div>
    </div>
  </div>

</div>

<script>
(function(){
  const el = (id) => document.getElementById(id);
  const siteInput = el('siteInput'), startBtn = el('startBtn'), stopBtn = el('stopBtn');
  const statusText = el('statusText'), banner = el('banner');
  const crawlBody = el('crawlBody'), linksSample = el('linksSample');
  const contextPill = el('contextPill'), sentenceText = el('sentenceText'), sentenceMeta = el('sentenceMeta');
  const progressBar = el('progressBar'), knowledgeView = el('knowledgeView'), logEl = el('log');
  const latencyCanvas = el('latencyChart'), progressCanvas = el('progressChart');
  const extractedBoxes = el('extractedBoxes'), extractedCount = el('extractedCount');
  let extractedEmpty = el('extractedEmpty');
  const nodeGraph = el('nodeGraph'), legendEl = el('legend');

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
  let hitMissPoints = [];        // parallel to progressPoints: true = sentence yielded a fact
  let sentenceHistory = [];      // {index, context, contributedKeys, chunkPreview}
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
    const row = document.createElement('div');
    row.className = 'row';
    row.textContent = text;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setBanner(kind, text){
    if (!text){ banner.innerHTML = ''; return; }
    banner.innerHTML = '<div class="msg-' + kind + '">' + text + '</div>';
  }

  function contextKind(context){
    if (!context) return 'paragraph';
    if (context.indexOf('bullet') === 0) return 'bullet';
    if (context === 'heading') return 'heading';
    return 'paragraph';
  }

  function pillClass(context){ return contextKind(context); }

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

  function drawProgressWithHits(canvas, values, hits, color){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    for (let i=1;i<4;i++){
      const y = h - (h*i/4);
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
    }
    // hit/miss strip along the bottom: green = useful info found, grey = just another sentence
    if (hits.length){
      const bw = w / hits.length;
      hits.forEach((hit, i) => {
        ctx.globalAlpha = hit ? 0.9 : 0.55;
        ctx.fillStyle = hit ? '#1a7f37' : '#d4d4d4';
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

  function redrawCharts(){
    drawLine(latencyCanvas, latencyPoints, '#1a1a1a');
    drawProgressWithHits(progressCanvas, progressPoints, hitMissPoints, '#1d4ed8');
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

  function svgEsc(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function buildChunkPreview(delta){
    const keys = computeContributedKeys(delta);
    if (!keys.length) return null;
    const k = keys[0];
    let v = delta[k];
    let first = Array.isArray(v) ? v[0] : v;
    let text = typeof first === 'string' ? first : JSON.stringify(first);
    text = (text || '').trim();
    if (text.length > 46) text = text.slice(0, 43) + '…';
    return { key: k, text: text || '(added)' };
  }

  function addExtractedBox(index, chunk){
    if (extractedEmpty){ extractedEmpty.remove(); extractedEmpty = null; }
    const color = CATEGORY_COLORS[chunk.key] || '#191919';
    const box = document.createElement('div');
    box.className = 'extracted-box';
    box.style.borderColor = color;
    box.innerHTML =
      '<div class="exb-head" style="color:' + color + '">' + svgEsc(CATEGORY_LABELS[chunk.key] || chunk.key) +
      '<span class="exb-idx">S' + (index+1) + '</span></div>' +
      '<div class="exb-text">' + svgEsc(chunk.text) + '</div>';
    extractedBoxes.appendChild(box);
    extractedCount.textContent = '(' + extractedBoxes.children.length + ')';
    extractedBoxes.scrollTop = extractedBoxes.scrollHeight;
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

    // wires: extracted-data chunk box -> the sentence node that produced it (new)
    visible.forEach((s, i) => {
      if (!s.chunkPreview) return;
      const sy = sentGap * (i+1);
      const x1 = chunkX + chunkW, y1 = sy, x2 = sentX, y2 = sy;
      const mx = (x1+x2)/2;
      const d = 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 + ', ' + x2 + ' ' + y2;
      const color = CATEGORY_COLORS[s.chunkPreview.key] || '#999';
      parts.push('<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.4" opacity="0.45"/>');
    });

    // wires (drawn first, under the nodes) -- sentence -> category, kept as-is
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

    // category nodes (right side)
    CATEGORY_ORDER.forEach(cat => {
      const p = catPos[cat], color = CATEGORY_COLORS[cat];
      parts.push(
        '<rect x="' + p.x + '" y="' + (p.y-14) + '" width="118" height="28" rx="8" fill="#ffffff" stroke="' + color + '" stroke-width="1.6"/>' +
        '<circle cx="' + p.x + '" cy="' + p.y + '" r="4" fill="' + color + '"/>' +
        '<text x="' + (p.x+12) + '" y="' + (p.y+4) + '" font-size="11" fill="' + color + '">' + svgEsc(CATEGORY_LABELS[cat]) + '</text>'
      );
    });

    // sentence nodes (left side)
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

    // extracted-data chunk nodes (far left, new) -- only sentences that hit
    visible.forEach((s, i) => {
      if (!s.chunkPreview) return;
      const sy = sentGap * (i+1);
      const color = CATEGORY_COLORS[s.chunkPreview.key] || '#999';
      parts.push(
        '<rect x="' + chunkX + '" y="' + (sy-15) + '" width="' + chunkW + '" height="30" rx="8" fill="#fbfbfb" stroke="' + color + '" stroke-width="1.4"/>' +
        '<text x="' + (chunkX+8) + '" y="' + (sy-2) + '" font-size="8.5" fill="' + color + '" font-weight="600">' + svgEsc(CATEGORY_LABELS[s.chunkPreview.key] || '') + '</text>' +
        '<text x="' + (chunkX+8) + '" y="' + (sy+9) + '" font-size="8.5" fill="#555">' + svgEsc(s.chunkPreview.text) + '</text>'
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

  async function pollStatus(){
    try{
      const r = await fetch('/api/status');
      const d = await r.json();
      statusText.textContent = d.connected
        ? ('connected · model ' + d.model + (d.models && d.models.length ? ' · ' + d.models.length + ' model(s) on host' : ''))
        : ('Ollama unreachable — ' + (d.error || 'unknown error'));
    }catch(e){
      statusText.textContent = 'status check failed';
    }
  }
  pollStatus();
  setInterval(pollStatus, 8000);

  function resetPanels(){
    setBanner(null, '');
    crawlBody.textContent = 'Crawling…';
    crawlBody.classList.remove('empty');
    linksSample.textContent = '';
    contextPill.innerHTML = '';
    sentenceText.textContent = '—';
    sentenceText.classList.add('empty');
    sentenceMeta.textContent = '';
    progressBar.value = 0; progressBar.max = 1;
    knowledgeView.textContent = '{}';
    logEl.innerHTML = '';
    latencyPoints = []; progressPoints = []; hitMissPoints = []; sentenceHistory = [];
    extractedBoxes.innerHTML = '<div class="empty" id="extractedEmpty">Nothing extracted yet — a box appears here every time a sentence yields a useful fact.</div>';
    extractedEmpty = el('extractedEmpty');
    extractedCount.textContent = '';
    redrawCharts(); renderGraph();
  }

  startBtn.addEventListener('click', async () => {
    const site = siteInput.value.trim();
    if (!site){ alert('Enter a site first, e.g. princeton.edu'); return; }
    resetPanels();
    startBtn.disabled = true; stopBtn.disabled = false;
    try{
      const r = await fetch('/api/start', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({site: site})
      });
      if (!r.ok){
        const d = await r.json().catch(()=>({}));
        setBanner('error', 'Could not start: ' + (d.error || r.status));
        startBtn.disabled = false; stopBtn.disabled = true;
      }
    }catch(e){
      setBanner('error', 'Could not start: ' + e.message);
      startBtn.disabled = false; stopBtn.disabled = true;
    }
  });

  stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    await fetch('/api/stop', {method:'POST'});
  });

  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    let evt;
    try{ evt = JSON.parse(e.data); } catch(err){ return; }
    handleEvent(evt);
  };
  es.onerror = () => { /* browser auto-reconnects SSE */ };

  function handleEvent(evt){
    switch(evt.event){
      case 'crawl_start': {
        logRow('[' + evt.ts + '] crawling ' + evt.site);
        break;
      }
      case 'retry':
        setBanner('error', 'network hiccup on ' + evt.what + ' — retrying (attempt ' + evt.attempt + '/' + evt.max_retries + ' failed: ' + evt.message + '), waiting ' + evt.wait + 's…');
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
        break;
      case 'page_extracted':
        crawlBody.textContent = 'Extracted ' + evt.item_count + ' sentence(s) from ' + evt.url;
        progressBar.max = evt.item_count || 1;
        logRow('[' + evt.ts + '] page walked -> ' + evt.item_count + ' sentence(s)');
        break;
      case 'sentence_start': {
        const cls = pillClass(evt.context);
        contextPill.innerHTML = '<span class="pill ' + cls + '">' + evt.context + '</span>';
        sentenceText.textContent = evt.text;
        sentenceText.classList.remove('empty');
        sentenceMeta.textContent = 'sentence ' + (evt.index+1) + ' / ' + evt.total + ' — asking qwen…';
        sentenceHistory.push({index: evt.index, context: evt.context, contributedKeys: []});
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

        if (entry.chunkPreview) addExtractedBox(evt.index, entry.chunkPreview);
        break;
      }
      case 'run_done':
        setBanner('good', evt.stopped_early ? 'Stopped early — knowledge collected so far is shown above.'
                                             : 'Done — knowledge JSON above is the full result.');
        logRow('[' + evt.ts + '] run finished' + (evt.stopped_early ? ' (stopped early)' : ''));
        startBtn.disabled = false; stopBtn.disabled = true;
        break;
      case 'stopped':
        logRow('[' + evt.ts + '] stopped by user');
        startBtn.disabled = false; stopBtn.disabled = true;
        break;
      case 'error':
        setBanner('error', evt.message);
        logRow('[' + evt.ts + '] ERROR: ' + evt.message);
        startBtn.disabled = false; stopBtn.disabled = true;
        break;
    }
  }

  renderGraph();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "Deepship/2.0"

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

        if self.path == "/api/start":
            site = (body.get("site") or "").strip()
            if RUN_STATE["running"]:
                self._send(409, json.dumps({"error": "a run is already in progress"}), "application/json")
                return
            threading.Thread(target=run_pipeline, args=(site,), daemon=True).start()
            self._send(200, json.dumps({"ok": True}), "application/json")
        elif self.path == "/api/stop":
            RUN_STATE["stop_flag"] = True
            self._send(200, json.dumps({"ok": True}), "application/json")
        else:
            self._send(404, "not found")

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
# Headless CLI (writes a JSON file, no browser)
# ---------------------------------------------------------------------------

def _run_cli(site, output_path):
    done = threading.Event()
    result = {}

    def listener():
        q = bus.subscribe()
        while True:
            payload = q.get()
            evt = json.loads(payload)
            name = evt.get("event")
            if name == "sentence_done":
                print(f"  [{evt['index']+1}/{evt['total']}] ({evt['elapsed']}s) "
                      f"context={evt.get('delta')}")
            elif name in ("crawl_start", "crawl_done", "triage_fallback",
                          "candidate_found", "page_extracted"):
                extra = {k: v for k, v in evt.items() if k not in ("event", "ts")}
                print(f"-- {name}: {extra}")
            elif name == "run_done":
                result["knowledge"] = evt.get("knowledge")
                done.set()
                return
            elif name in ("stopped", "error"):
                if name == "error":
                    print(f"! error: {evt.get('message')}", file=sys.stderr)
                done.set()
                return

    threading.Thread(target=listener, daemon=True).start()
    run_pipeline(site)
    done.wait()

    if result.get("knowledge"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["knowledge"], f, indent=2, ensure_ascii=False)
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
                     "learn what it says one sentence at a time via a local qwen2.5 Ollama "
                     "server. Run with no arguments to launch the live web GUI instead."
    )
    parser.add_argument("site", nargs="?", help="College domain, e.g. princeton.edu (CLI mode)")
    parser.add_argument("--output", default="deepship_result.json", help="Where to write JSON (CLI mode)")
    parser.add_argument("--port", type=int, default=8765, help="Port for the web GUI (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    args = parser.parse_args()

    if args.site:
        _run_cli(args.site, args.output)
    else:
        run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()