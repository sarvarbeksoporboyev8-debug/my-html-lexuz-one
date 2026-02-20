#!/usr/bin/env python3
"""
Legal Q&A using Perplexity + Gemini
Cost: ~$0.001 per query

Usage:
    python search_ask.py "QQS stavkasi necha foiz?"
    python search_ask.py "Mehnat kodeksida yillik ta'til muddati"
"""

import os
import sys
import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs, unquote, quote
from bs4 import BeautifulSoup

# Search provider priority: 1=DuckDuckGo (web), 2=Gemini, 3=Perplexity
# Perplexity API (third priority / fallback)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# Google Gemini API with Search Grounding (second priority)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Restrict DuckDuckGo to site:lex.uz when enabled (see WEB_SEARCH_SITE_FILTER_ENABLED below)
WEB_SEARCH_SITE_FILTER = "site:lex.uz"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


LOW_COST_MODE = _env_bool("LOW_COST_MODE", True)

# Hard caps for predictable spend
WEB_MAX_RESULTS = _env_int("WEB_MAX_RESULTS", 4 if LOW_COST_MODE else 10)
WEB_TITLE_MAX_CHARS = _env_int("WEB_TITLE_MAX_CHARS", 140 if LOW_COST_MODE else 180)
WEB_SNIPPET_MAX_CHARS = _env_int("WEB_SNIPPET_MAX_CHARS", 220 if LOW_COST_MODE else 400)
# Context sent to Gemini: ~3.5k (low-cost) or ~32k so 20 results + 5 full-page excerpts can fit
WEB_CONTEXT_MAX_CHARS = _env_int("WEB_CONTEXT_MAX_CHARS", 3500 if LOW_COST_MODE else 32000)
# Optional: fetch full page content for top N results so Gemini reads actual articles (deeper answers)
WEB_FETCH_FULL_PAGES = _env_bool("WEB_FETCH_FULL_PAGES", False)
WEB_FETCH_TOP_N = _env_int("WEB_FETCH_TOP_N", 5)
WEB_FETCH_MAX_CHARS_PER_PAGE = _env_int("WEB_FETCH_MAX_CHARS_PER_PAGE", 2500)
# DuckDuckGo rate-limit avoidance: retries and delay (seconds)
WEB_SEARCH_RETRIES = _env_int("WEB_SEARCH_RETRIES", 3)
WEB_SEARCH_RETRY_DELAY = _env_int("WEB_SEARCH_RETRY_DELAY", 2)
WEB_SEARCH_INITIAL_DELAY = _env_int("WEB_SEARCH_INITIAL_DELAY", 1)
# Debug: save and/or print DuckDuckGo HTML when 200 but 0 results (to inspect CAPTCHA vs no-results page)
WEB_DEBUG_SAVE_HTML = _env_bool("WEB_DEBUG_SAVE_HTML", False)
WEB_DEBUG_SAVE_HTML_PATH = os.getenv("WEB_DEBUG_SAVE_HTML_PATH", "debug_duckduckgo_last.html")
WEB_DEBUG_PRINT_HTML = _env_bool("WEB_DEBUG_PRINT_HTML", False)  # print truncated response to terminal (e.g. in railway ssh)
WEB_DEBUG_PRINT_HTML_MAX = _env_int("WEB_DEBUG_PRINT_HTML_MAX", 4000)
# Bright Data proxy for DuckDuckGo (avoids CAPTCHA); set BRIGHTDATA_ENABLED=1 and host/port/user/pass in Railway
BRIGHTDATA_ENABLED = _env_bool("BRIGHTDATA_ENABLED", False)
# When False, DuckDuckGo search is not restricted to site:lex.uz (broader results; Gemini still uses sources)
WEB_SEARCH_SITE_FILTER_ENABLED = _env_bool("WEB_SEARCH_SITE_FILTER_ENABLED", False)
GEMINI_WEB_SUMMARY_MAX_OUTPUT_TOKENS = _env_int("GEMINI_WEB_SUMMARY_MAX_OUTPUT_TOKENS", 600 if LOW_COST_MODE else 1200)
PERPLEXITY_REWRITE_MAX_TOKENS = _env_int("PERPLEXITY_REWRITE_MAX_TOKENS", 80 if LOW_COST_MODE else 100)
PERPLEXITY_ANSWER_MAX_TOKENS = _env_int("PERPLEXITY_ANSWER_MAX_TOKENS", 1600 if LOW_COST_MODE else 4000)



def extract_title_from_url(url: str) -> str:
    """Extract a readable title from lex.uz URL."""
    # Try to get doc ID from URL
    import re
    match = re.search(r'/docs/(-?\d+)', url)
    if match:
        doc_id = match.group(1)
        return f"Lex.uz hujjat #{doc_id}"
    return "Lex.uz hujjat"


def extract_domain_label(url: str) -> str:
    """Extract short label from URL domain (e.g., lex.uz -> lex, xabar.uz -> xabar)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        
        # Remove www. prefix
        if host.startswith("www."):
            host = host[4:]
        
        # Remove common TLDs
        tlds = [".uz", ".com", ".ru", ".org", ".net", ".io", ".co", ".info"]
        for tld in tlds:
            if host.endswith(tld):
                host = host[:-len(tld)]
                break
        
        # Remove subdomains - keep only main domain part
        parts = host.split(".")
        if len(parts) > 1:
            host = parts[-1]  # Take last part (main brand)
        
        # Truncate if too long
        if len(host) > 10:
            host = host[:10]
        
        return host or "web"
    except:
        return "web"


def _search_web_ddgs(query: str, max_results: int) -> list:
    """Primary DuckDuckGo path: use ddgs package (avoids CAPTCHA/HTML parsing)."""
    try:
        from ddgs import DDGS
        proxies = _get_brightdata_proxies()
        proxy_url = proxies.get("https") or proxies.get("http") if proxies else None
        if proxy_url:
            print("[WEB] Using Bright Data proxy for DuckDuckGo.")
        client = DDGS(proxy=proxy_url, timeout=30) if proxy_url else DDGS(timeout=30)
        raw = client.text(query, max_results=max_results, backend="duckduckgo")
        results = list(raw) if raw else []
    except ImportError:
        return []
    except Exception as e:
        print(f"[WEB] ddgs failed: {e}")
        return []
    if not results:
        return []
    out = []
    seen = set()
    for i, r in enumerate(results[:max_results], 1):
        url = r.get("href") or r.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        title = (r.get("title") or "")[:WEB_TITLE_MAX_CHARS]
        snippet = (r.get("body") or r.get("snippet") or "")[:WEB_SNIPPET_MAX_CHARS]
        out.append({
            "id": i,
            "title": title or extract_title_from_url(url),
            "url": url,
            "snippet": snippet,
            "label": extract_domain_label(url),
        })
    if out:
        print(f"[WEB] DuckDuckGo (ddgs): {len(out)} results.")
    return out


# Headers that look like a real browser to reduce DuckDuckGo rate limiting / 202
_DUCKDUCKGO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://duckduckgo.com/",
    "Upgrade-Insecure-Requests": "1",
}


def _normalize_ddg_url(href: str) -> str:
    """Extract real URL from DuckDuckGo redirect (uddg=...) or return as-is."""
    if not href:
        return ""
    if "uddg=" in href:
        try:
            parsed = parse_qs(urlparse(href).query)
            return unquote(parsed.get("uddg", [href])[0])
        except Exception:
            pass
    return href


def _parse_duckduckgo_html(soup: "BeautifulSoup", effective_max_results: int) -> list:
    """Parse DuckDuckGo HTML with several selector strategies (DDG changes class names over time)."""
    results = []
    seen_urls = set()

    def add_result(link_el, snippet_el=None):
        nonlocal results, seen_urls
        url = link_el.get("href", "").strip()
        url = _normalize_ddg_url(url)
        if not url or url in seen_urls:
            return False
        title = link_el.get_text(" ", strip=True)
        snippet = ""
        if snippet_el:
            snippet = snippet_el.get_text(" ", strip=True)
        seen_urls.add(url)
        results.append({
            "id": len(results) + 1,
            "title": (title or extract_title_from_url(url))[:WEB_TITLE_MAX_CHARS],
            "url": url,
            "snippet": snippet[:WEB_SNIPPET_MAX_CHARS],
            "label": extract_domain_label(url),
        })
        return True

    # Strategy 1: .result + a.result__a (legacy)
    for block in soup.select(".result"):
        link_el = block.select_one("a.result__a")
        if not link_el:
            link_el = block.select_one("a.result__url")
        if not link_el:
            continue
        snippet_el = block.select_one(".result__snippet")
        if add_result(link_el, snippet_el) and len(results) >= effective_max_results:
            return results

    # Strategy 2: any a.result__url or a[class*="result__"] with uddg (DDG HTML variant)
    if not results:
        for a in soup.select('a[href*="uddg="]'):
            if a.get("class") and any("result" in c for c in a.get("class", [])):
                parent = a.find_parent(class_=re.compile(r"result", re.I))
                snip = parent.select_one(".result__snippet") if parent else None
                if add_result(a, snip) and len(results) >= effective_max_results:
                    return results

    # Strategy 3: links in .results_links or similar container
    if not results:
        for block in soup.select(".results_links, .results_links_deep, [class*='result']"):
            for link in block.select('a[href*="uddg="]'):
                if add_result(link, None) and len(results) >= effective_max_results:
                    return results

    return results


def _get_brightdata_proxies():
    """Build Bright Data proxy dict from env (BRIGHTDATA_ENABLED, HOST, PORT, USERNAME, PASSWORD)."""
    if not BRIGHTDATA_ENABLED:
        return None
    host = os.getenv("BRIGHTDATA_HOST", "").strip()
    port = os.getenv("BRIGHTDATA_PORT", "33335").strip()
    user = os.getenv("BRIGHTDATA_USERNAME", "").strip()
    password = os.getenv("BRIGHTDATA_PASSWORD", "").strip()
    if not host or not user or not password:
        return None
    # http://user:pass@host:port (quote user/pass in case of special chars)
    auth = f"{quote(user, safe='')}:{quote(password, safe='')}"
    proxy_url = f"http://{auth}@{host}:{port}"
    return {"http": proxy_url, "https": proxy_url}


def search_web_top_results(query: str, max_results: int = 8) -> list:
    """Search DuckDuckGo: primary = duckduckgo-search package; fallback = HTML scrape."""
    try:
        effective_max_results = max(1, min(max_results, WEB_MAX_RESULTS))
        final_query = query.strip()
        if WEB_SEARCH_SITE_FILTER_ENABLED and WEB_SEARCH_SITE_FILTER not in final_query:
            final_query = f"{WEB_SEARCH_SITE_FILTER} {final_query}".strip()

        # 1. Primary: duckduckgo-search package (avoids CAPTCHA, no HTML parsing)
        results = _search_web_ddgs(final_query, effective_max_results)
        if results:
            return results

        # 2. Fallback: HTML scrape (retries, Bright Data, multiple selectors)
        print("[WEB] Falling back to DuckDuckGo HTML scrape.")
        if WEB_SEARCH_INITIAL_DELAY > 0:
            time.sleep(WEB_SEARCH_INITIAL_DELAY)
        url = "https://duckduckgo.com/html/"
        params = {"q": final_query}
        last_error_status = None
        resp = None

        for attempt in range(WEB_SEARCH_RETRIES):
            if attempt > 0:
                delay = WEB_SEARCH_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[WEB] Retry in {delay}s (attempt {attempt + 1}/{WEB_SEARCH_RETRIES})...")
                time.sleep(delay)

            proxies = _get_brightdata_proxies()
            if attempt == 0 and proxies:
                print("[WEB] Using Bright Data proxy for DuckDuckGo.")
            resp = requests.get(
                url,
                params=params,
                headers=_DUCKDUCKGO_HEADERS,
                proxies=proxies,
                timeout=35,
            )
            last_error_status = resp.status_code

            if resp.status_code == 200:
                break
            if resp.status_code not in (202, 429, 503):
                print(f"[WEB] Search failed: {resp.status_code}")
                return []
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
                print(f"[WEB] Search {resp.status_code}, Retry-After={delay}s")
                time.sleep(delay)
            else:
                print(f"[WEB] Search failed: {resp.status_code}")
        else:
            print(f"[WEB] Search failed after {WEB_SEARCH_RETRIES} attempts: {last_error_status}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results = _parse_duckduckgo_html(soup, effective_max_results)

        if not results:
            print("[WEB] DuckDuckGo HTML returned no results (CAPTCHA or empty page).")
            if WEB_DEBUG_SAVE_HTML:
                try:
                    path = os.path.abspath(WEB_DEBUG_SAVE_HTML_PATH)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"<!-- query: {final_query!r}\n")
                        f.write(f"     saved: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} -->\n")
                        f.write(resp.text)
                    print(f"[WEB] Debug: saved response to {path}")
                except Exception as e:
                    print(f"[WEB] Debug save failed: {e}")
            if WEB_DEBUG_PRINT_HTML or WEB_DEBUG_SAVE_HTML:
                text = resp.text
                hints = []
                tlower = text.lower()
                if "captcha" in tlower or "recaptcha" in tlower or "challenge" in tlower:
                    hints.append("captcha/challenge")
                if "no results" in tlower or "no result" in tlower:
                    hints.append("no results")
                if "verify" in tlower and ("human" in tlower or "browser" in tlower):
                    hints.append("verify human/browser")
                print("[WEB] Debug hints:", ", ".join(hints) if hints else "none")
                snippet = text[:WEB_DEBUG_PRINT_HTML_MAX]
                if len(text) > WEB_DEBUG_PRINT_HTML_MAX:
                    snippet += "\n... [truncated, total %d chars]" % len(text)
                print("--- DuckDuckGo response (first %d chars) ---" % len(snippet))
                print(snippet)
                print("--- end ---")

        return results

    except Exception as e:
        print(f"[WEB] Search exception: {e}")
        return []


def _merge_web_results(first: list, second: list, prefer_uz: bool = True) -> list:
    """Merge two result lists, dedupe by URL. Optionally put lex.uz / .uz sources first."""
    seen = set()
    out = []
    def add(item):
        u = item.get("url", "")
        if not u or u in seen:
            return
        seen.add(u)
        out.append(item)
    # Prefer Uzbek/lex.uz domains first if requested
    def is_uz_source(item):
        u = (item.get("url") or "").lower()
        try:
            host = urlparse(u).netloc
            return "lex.uz" in u or "gov.uz" in u or host.endswith(".uz")
        except Exception:
            return "lex.uz" in u or ".uz" in u
    if prefer_uz:
        for item in first + second:
            if is_uz_source(item):
                add(item)
        for item in first + second:
            if not is_uz_source(item):
                add(item)
    else:
        for item in first + second:
            add(item)
    for i, item in enumerate(out, 1):
        item["id"] = i
    return out


def _fetch_page_text(url: str, max_chars: int, timeout: int = 10) -> str:
    """Fetch URL and return plain text (body), truncated to max_chars. Returns '' on failure."""
    try:
        proxies = _get_brightdata_proxies()
        resp = requests.get(
            url,
            headers={"User-Agent": _DUCKDUCKGO_HEADERS.get("User-Agent", "Mozilla/5.0")},
            proxies=proxies,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars] if text else ""
    except Exception:
        return ""


def rewrite_query_with_perplexity(question: str) -> str:
    """Rewrite user question into a concise web-search query using Perplexity."""
    if not PERPLEXITY_API_KEY:
        return None

    system_prompt = """You output a single search query for Google/DuckDuckGo. Rules:
- One line only, no explanation. No quotes or prefixes.
- Use key phrases and keywords, not full sentences. 6–12 words.
- Include "O'zbekiston" (or Uzbekistan) when the question is about Uzbekistan to get local sources.
- Use terms that appear on official sources (e.g. qonun, tartib, qoidalar, ruxsat, davlat organlari, lex.uz) as appropriate to the question topic."""

    site_hint = "\nInclude site:lex.uz in the query.\n" if WEB_SEARCH_SITE_FILTER_ENABLED else ""
    user_prompt = f"""Write one search query that will find the best results for this question. Output only the query.{site_hint}

Question: {question}"""

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": PERPLEXITY_REWRITE_MAX_TOKENS,
            },
            timeout=20,
        )

        if resp.status_code != 200:
            print(f"[PERPLEXITY-REWRITE] Error: {resp.status_code}")
            return None

        data = resp.json()
        rewritten = data["choices"][0]["message"]["content"].strip()
        rewritten = rewritten.split("\n")[0].strip().strip('"').strip("'")
        # Trim trailing question markers so the query is more keyword-like for search
        while rewritten and rewritten[-1] in "?؟":
            rewritten = rewritten[:-1].strip()
        if rewritten and WEB_SEARCH_SITE_FILTER_ENABLED and WEB_SEARCH_SITE_FILTER not in rewritten:
            rewritten = f"{WEB_SEARCH_SITE_FILTER} {rewritten}".strip()
        return rewritten if rewritten else None

    except Exception as e:
        print(f"[PERPLEXITY-REWRITE] Failed: {e}")
        return None


def rewrite_queries_with_perplexity(question: str, max_queries: int = 5) -> list:
    """Ask Perplexity for multiple targeted search queries (like Google AI's approach). Returns list of query strings."""
    if not PERPLEXITY_API_KEY:
        return []

    system_prompt = """You are a search strategist. Output 4–5 different Google/DuckDuckGo search queries that together will find the best information for the user's question. The topic is whatever the user asks about (do not assume crypto, finance, or any specific domain). Vary the angle: e.g. relevant law/regulation, specific entities or bodies mentioned, restrictions or permissions, official sources (lex.uz, gov.uz, NAPP when relevant). Use Uzbek and English keywords appropriate to the question. One query per line. No numbering, bullets, or explanation. 6–12 words per query."""

    user_prompt = f"""Question: {question}

Output 4–5 search queries (one per line) that would cross-reference the right laws, entities, and official sources to answer this question accurately. Base the queries only on the question above."""

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": min(220, PERPLEXITY_REWRITE_MAX_TOKENS * 2),
            },
            timeout=25,
        )
        if resp.status_code != 200:
            print(f"[PERPLEXITY-REWRITE] Error: {resp.status_code}")
            return []
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        lines = [ln.strip().strip('"\'').strip() for ln in text.split("\n") if ln.strip()]
        out = []
        for ln in lines:
            while ln and ln[-1] in "?؟":
                ln = ln[:-1].strip()
            if not ln or len(ln) < 4:
                continue
            if WEB_SEARCH_SITE_FILTER_ENABLED and WEB_SEARCH_SITE_FILTER not in ln:
                ln = f"{WEB_SEARCH_SITE_FILTER} {ln}".strip()
            out.append(ln)
            if len(out) >= max_queries:
                break
        return out
    except Exception as e:
        print(f"[PERPLEXITY-REWRITE] Failed: {e}")
        return []


def summarize_web_results_with_gemini(question: str, results: list) -> str:
    """Summarize fetched web results with Gemini, citing [1], [2], ... markers.
    If WEB_FETCH_FULL_PAGES is True, fetches full page content for top N results in parallel so Gemini can read actual articles.
    """
    if not GEMINI_API_KEY or not results:
        return None

    to_process = results[:WEB_MAX_RESULTS]
    # Pre-fetch full page text in parallel (avoids 5 × 10s sequential wait)
    fetched = {}
    if WEB_FETCH_FULL_PAGES and WEB_FETCH_TOP_N > 0:
        to_fetch = [(idx, item) for idx, item in enumerate(to_process) if idx < WEB_FETCH_TOP_N and item.get("url", "").startswith("http")]
        if to_fetch:
            print(f"[WEB] Fetching {len(to_fetch)} full page(s) in parallel (timeout 10s each)...")
            def fetch_one(args):
                idx, item = args
                url = item.get("url", "")
                return idx, _fetch_page_text(url, WEB_FETCH_MAX_CHARS_PER_PAGE, timeout=10)
            with ThreadPoolExecutor(max_workers=min(5, len(to_fetch))) as ex:
                for result in ex.map(fetch_one, to_fetch):
                    idx, text = result
                    if text:
                        fetched[idx] = text
            if fetched:
                print(f"[WEB] Fetched {len(fetched)} page(s) for deeper context.")

    context_lines = []
    used_chars = 0
    for idx, item in enumerate(to_process):
        line = f"[{item['id']}] {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('snippet', '')}"
        if idx in fetched and fetched[idx]:
            line += f"\nFull text (excerpt): {fetched[idx]}"
        if used_chars + len(line) > WEB_CONTEXT_MAX_CHARS:
            line = line[: max(0, WEB_CONTEXT_MAX_CHARS - used_chars - 50)] + "\n..."
        context_lines.append(line)
        used_chars += len(line)
        if used_chars >= WEB_CONTEXT_MAX_CHARS:
            break

    prompt = f"""Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

SAVOL:
{question}

INTERNET MANBALARI (top natijalar):
{chr(10).join(context_lines)}

QOIDALAR:
1. Faqat berilgan manbalarga tayangan holda javob bering. Agar manbada "Full text (excerpt)" bo'lsa, shu matn asosida aniq va chuqur javob bering.
2. O'zbek tilida yozing, 2-5 paragraf, aniq va asoslangan (snippetdan ko'ra to'liq matn bo'lsa batafsilroq yozing).
3. Aniq, ishonchli tonda yozing. "Balki", "ehtimol", "tavsiya etiladi" kabi noaniq ifodalar ishlatmang; manbalarga asoslanib aniq xulosa bering.
4. Har paragraf oxirida [1], [2] kabi iqtibos raqamlarini qo'ying.
5. Taxmin qilmang; manbada bo'lmasa ochiq ayting.
6. Oxirida quyidagi bo'limlarni yozing:

Manbalar:
[N] Sarlavha - URL

Tegishli savollar:
- 5 ta tegishli savol
"""

    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    for model in models:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": GEMINI_WEB_SUMMARY_MAX_OUTPUT_TOKENS,
                    },
                },
                timeout=90,
            )

            if resp.status_code == 429:
                print(f"[WEB+GEMINI] {model} rate limited, trying next model...")
                continue

            if resp.status_code != 200:
                print(f"[WEB+GEMINI] {model} failed: {resp.status_code}")
                continue

            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            print(f"[WEB+GEMINI] {model} exception: {e}")
            continue

    return None


def build_web_results_fallback_answer(results: list) -> str:
    """Build plain fallback answer from web search results without an LLM."""
    if not results:
        return "Internetdan mos natijalar topilmadi."

    lines = [
        "Perplexity/Gemini orqali to'liq javob olib bo'lmadi. Quyida internetdagi eng mos topilgan manbalar:",
        "",
        "Manbalar:",
    ]

    for item in results:
        title = item.get("title", "Manba")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        lines.append(f"[{item['id']}] {title} - {url}")
        if snippet:
            lines.append(f"    {snippet}")

    lines += [
        "",
        "Tegishli savollar:",
        "- Shu mavzuga oid aniq normativ hujjat qaysi?",
        "- Bu sohada qanday cheklovlar yoki talablar mavjud?",
        "- Qaysi davlat organi bu masalani nazorat qiladi?",
        "- Tegishli qonun yoki qoidalar qachon kuchga kirgan?",
        "- Amaliyotda qanday qo'llaniladi?",
    ]
    return "\n".join(lines)


def build_perplexity_multisearch_question(question: str) -> str:
    """Build a Perplexity prompt that forces multiple targeted search passes. Topic-agnostic."""
    return f"""O'zbekiston qonunchiligi bo'yicha savol: {question}

QIDIRUV STRATEGIYASI (MAJBURIY):
1) Kamida 3-4 ta alohida qidiruv o'tkazing (turli kalit so'zlar bilan)
2) lex.uz va rasmiy manbalarga tayaning
3) Savol mavzusiga qarab turli yo'nalishlarda qidiring: tegishli qonun/hujjatlar, cheklovlar yoki ruxsatlar, davlat organlari vakolati, nazorat talablari. Mavzuga mos kalit so'zlardan foydalaning.

JAVOB QOIDALARI:
- O'zbek tilida 2-4 paragraf aniq xulosa bering
- Har paragrafda [1], [2] kabi citation bo'lsin
- Agar to'g'ridan-to'g'ri norma topilmasa, buni aniq yozing va qaysi normativ bo'shliq borligini ayting
- Faqat manbalarda bor ma'lumotni yozing, taxmin qilmang
"""


def ask_gemini_structured(question: str, history: list = None) -> dict:
    """Use Google Gemini with Search Grounding, return structured response."""
    
    if not GEMINI_API_KEY:
        return None
    
    # Build conversation contents
    contents = []
    
    # Add chat history if provided
    if history:
        for msg in history[-1:]:  # Last 1 message only
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("content", "")
            if text:
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })
    
    # Add current question with instructions for structured output
    full_question = f"""{question}

MUHIM:
- Javobni paragraflar bilan yoz. Har bir paragrafdan keyin qaysi manbadan olganingni [1], [2] kabi raqam bilan ko'rsat.
- O'zbek tilida javob ber. Qisqa va aniq (2-4 paragraf).
- Aniq va ishonchli tonda yozing. "Taxminan", "balki", "ehtimol", "tavsiya etiladi", "harakat qilaman" kabi noaniq yoki shubhali ifodalar ishlatmang. Manbalarga asoslanib aniq xulosa bering.
- Oxirida 5 ta tegishli savol yoz."""
    
    contents.append({
        "role": "user",
        "parts": [{"text": full_question}]
    })
    
    # Try multiple models in case of rate limits
    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]
    
    resp = None
    for model in models:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": contents,
                    "tools": [
                        {"google_search": {}}
                    ]
                },
                timeout=30,
            )
            
            if resp.status_code == 429:
                print(f"[GEMINI] {model} rate limited, trying next model...")
                time.sleep(1)
                continue
            
            if resp.status_code == 200:
                break
                
        except Exception as e:
            print(f"[GEMINI] {model} failed: {e}")
            continue
    else:
        return None
    
    try:
        if resp.status_code != 200:
            print(f"[GEMINI] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        
        # Extract answer text
        answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract sources from grounding metadata
        grounding = data["candidates"][0].get("groundingMetadata", {})
        chunks = grounding.get("groundingChunks", [])
        
        # Build sources dict (id -> source object)
        sources = {}
        seen_urls = set()
        source_id = 1
        
        for chunk in chunks:
            url = chunk.get("web", {}).get("uri", "")
            title = chunk.get("web", {}).get("title", "") or ""
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                label = extract_domain_label(url)
                
                # Try to extract snippet from the chunk
                snippet = ""
                if "text" in chunk:
                    snippet = chunk["text"][:150]
                
                sources[str(source_id)] = {
                    "id": source_id,
                    "url": url,
                    "domain": urlparse(url).netloc if url else "",
                    "label": label,
                    "title": title[:100] if title else f"{label} hujjat",
                    "snippet": snippet
                }
                source_id += 1
        
        # Parse answer into blocks with anchorSources (ordered by relevance)
        blocks = parse_answer_to_blocks_v2(answer_text, sources)
        
        # Generate context-aware related questions
        related_questions = generate_related_questions(question, answer_text)
        
        return {
            "blocks": blocks,
            "sources": sources,
            "relatedQuestions": related_questions
        }
        
    except Exception as e:
        print(f"[GEMINI] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_answer_to_blocks_v2(text: str, sources: dict) -> list:
    """Parse answer text into blocks with anchorSources (Perplexity style)."""
    blocks = []
    
    # Remove related questions section from text
    text_clean = text
    for marker in ["Tegishli savollar:", "Related questions:", "Savollar:"]:
        if marker in text_clean:
            text_clean = text_clean.split(marker)[0]
    
    # Remove sources section if present
    for marker in ["Manbalar:", "Sources:", "Manba:"]:
        if marker in text_clean:
            text_clean = text_clean.split(marker)[0]
    
    # Remove numbered questions at the end (e.g., "1. Question? 2. Question?")
    # Pattern: starts with "1." and contains multiple numbered items with "?"
    numbered_questions_pattern = r'\n\s*1\.\s+[^\n]*\?\s*(?:\d+\.\s+[^\n]*\?\s*)+$'
    text_clean = re.sub(numbered_questions_pattern, '', text_clean, flags=re.DOTALL)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
    
    # If no double newlines, try single newlines for shorter responses
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text_clean.split("\n") if p.strip() and len(p.strip()) > 20]
    
    # Get all source IDs for distribution
    all_source_ids = list(sources.keys())
    
    for i, para in enumerate(paragraphs):
        if not para or len(para) < 10:
            continue
            
        # Find citation references like [1], [2], [3] in this paragraph
        citation_pattern = r'\[(\d+)\]'
        found_ids = list(dict.fromkeys(re.findall(citation_pattern, para)))  # Preserve order, remove dupes
        
        # Filter to only valid source IDs
        anchor_sources = [sid for sid in found_ids if sid in sources]
        
        # If no citations found in text, assign sources based on paragraph position
        if not anchor_sources and all_source_ids:
            # Distribute sources across paragraphs
            sources_per_para = max(1, len(all_source_ids) // max(1, len(paragraphs)))
            start_idx = i * sources_per_para
            end_idx = min(start_idx + sources_per_para, len(all_source_ids))
            if start_idx < len(all_source_ids):
                anchor_sources = all_source_ids[start_idx:end_idx]
            else:
                # Last paragraphs get remaining sources
                anchor_sources = all_source_ids[-1:] if all_source_ids else []
        
        # Clean the paragraph text (remove citation markers)
        clean_text = re.sub(r'\s*\[\d+\]\s*', ' ', para).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        if clean_text:
            blocks.append({
                "text": clean_text,
                "anchorSources": anchor_sources  # Ordered by relevance (first = primary)
            })
    
    # If no blocks created, create one from the whole text
    if not blocks and text_clean.strip():
        all_ids = list(dict.fromkeys(re.findall(r'\[(\d+)\]', text_clean)))
        anchor_sources = [sid for sid in all_ids if sid in sources]
        if not anchor_sources:
            anchor_sources = all_source_ids[:3] if all_source_ids else []
        clean_text = re.sub(r'\s*\[\d+\]\s*', ' ', text_clean).strip()
        blocks.append({
            "text": clean_text,
            "anchorSources": anchor_sources
        })
    
    return blocks


def parse_answer_to_blocks(text: str, sources: list) -> list:
    """Parse answer text into blocks with source IDs (legacy)."""
    blocks = []
    
    # Remove related questions section from text
    text_clean = text
    for marker in ["Tegishli savollar:", "Related questions:", "Savollar:"]:
        if marker in text_clean:
            text_clean = text_clean.split(marker)[0]
    
    # Remove sources section if present
    for marker in ["Manbalar:", "Sources:", "Manba:"]:
        if marker in text_clean:
            text_clean = text_clean.split(marker)[0]
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
    
    # If no double newlines, try single newlines for shorter responses
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text_clean.split("\n") if p.strip() and len(p.strip()) > 20]
    
    for para in paragraphs:
        if not para or len(para) < 10:
            continue
            
        # Find citation references like [1], [2], [3] in this paragraph
        citation_pattern = r'\[(\d+)\]'
        found_ids = list(set(int(m) for m in re.findall(citation_pattern, para)))
        
        # Filter to only valid source IDs
        valid_ids = [sid for sid in found_ids if any(s["id"] == sid for s in sources)]
        
        # Clean the paragraph text (remove citation markers for cleaner display)
        clean_text = re.sub(r'\s*\[\d+\]\s*', ' ', para).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        if clean_text:
            blocks.append({
                "text": clean_text,
                "sourceIds": valid_ids if valid_ids else []
            })
    
    # If no blocks created, create one from the whole text
    if not blocks and text_clean.strip():
        all_ids = list(set(int(m) for m in re.findall(r'\[(\d+)\]', text_clean)))
        valid_ids = [sid for sid in all_ids if any(s["id"] == sid for s in sources)]
        clean_text = re.sub(r'\s*\[\d+\]\s*', ' ', text_clean).strip()
        blocks.append({
            "text": clean_text,
            "sourceIds": valid_ids if valid_ids else ([1] if sources else [])
        })
    
    return blocks


def extract_related_questions(text: str) -> list:
    """Extract related questions from answer text."""
    questions = []
    
    # Find the related questions section
    markers = ["Tegishli savollar:", "Related questions:", "Savollar:", "Qo'shimcha savollar:"]
    section_text = ""
    
    for marker in markers:
        if marker.lower() in text.lower():
            idx = text.lower().find(marker.lower())
            section_text = text[idx + len(marker):]
            break
    
    if section_text:
        # Extract questions (lines starting with -, *, or numbers)
        lines = section_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove bullet points and numbers
            clean = re.sub(r'^[-•*\d.)\]]+\s*', '', line).strip()
            clean = clean.replace('**', '').strip()
            if clean and len(clean) > 5 and clean.endswith('?'):
                questions.append(clean)
            if len(questions) >= 6:
                break
    
    return questions


def generate_related_questions_with_gemini(question: str, answer_text: str) -> list:
    """Generate related questions using Gemini Flash (free tier)."""
    
    if not GEMINI_API_KEY:
        return []
    
    prompt = f"""Savol: {question}

Javob: {answer_text[:1000]}

Yuqoridagi savol va javobga asoslanib, O'zbekiston qonunchiligi bo'yicha 5 ta tegishli savol yozing.

QOIDALAR:
1. Har bir savol mustaqil bo'lsin - oldingi javobga bog'liq bo'lmasin
2. Savollar aniq va qisqa bo'lsin
3. Faqat savollarni yozing, boshqa hech narsa yozmang
4. Har bir savol yangi qatordan boshlansin
5. Savol oxirida ? belgisi bo'lsin

Savollar:"""

    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 300,
                }
            },
            timeout=15,
        )
        
        if resp.status_code != 200:
            print(f"[GEMINI] Error generating questions: {resp.status_code}")
            return []
        
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Parse questions from response
        questions = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering and bullets
            clean = re.sub(r'^[-•*\d.)\]]+\s*', '', line).strip()
            clean = clean.replace('**', '').strip()
            if clean and len(clean) > 10 and '?' in clean:
                # Extract just the question part
                if '?' in clean:
                    clean = clean[:clean.rfind('?') + 1]
                questions.append(clean)
            if len(questions) >= 5:
                break
        
        return questions
        
    except Exception as e:
        print(f"[GEMINI] Failed to generate questions: {e}")
        return []


def generate_related_questions(question: str, answer_text: str) -> list:
    """Generate related questions using Gemini Flash."""
    return generate_related_questions_with_gemini(question, answer_text)


def ask_gemini_grounded(question: str, history: list = None) -> str:
    """Use Google Gemini with Search Grounding (legacy string response)."""
    result = ask_gemini_structured(question, history)
    if not result:
        return None
    
    # Convert structured response back to string for backward compatibility
    answer_parts = []
    for block in result.get("blocks", []):
        text = block.get("text", "")
        source_ids = block.get("sourceIds", [])
        if source_ids:
            text += " " + " ".join(f"[{sid}]" for sid in source_ids)
        answer_parts.append(text)
    
    answer = "\n\n".join(answer_parts)
    
    # Add sources (sources is a dict id -> {id, title, url, ...})
    sources = result.get("sources", {})
    if sources:
        answer += "\n\nManbalar:\n"
        for s in sources.values():
            answer += f"[{s['id']}] {s['title']} - {s['url']}\n"
    
    # Add related questions
    questions = result.get("relatedQuestions", [])
    if questions:
        answer += "\n\nTegishli savollar:\n"
        for q in questions:
            answer += f"- {q}\n"
    
    return answer


def ask_perplexity_structured(question: str, history: list = None) -> dict:
    """Use Perplexity API, return structured response like Gemini."""
    
    if not PERPLEXITY_API_KEY:
        return None
    
    full_question = build_perplexity_multisearch_question(question)
    
    system_prompt = """Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

QOIDALAR:
1. Javoblarni o'zbek tilida, aniq va qisqa bering (2-4 paragraf)
2. Har bir paragrafda manbalarni [1], [2] kabi raqamlar bilan ko'rsating
3. HECH QACHON ** belgisini ishlatmang - bold/qalin matn kerak emas
4. Tegishli savollar YOZMA - men o'zim qo'shaman"""
    
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_question}
                ],
                "search_domain_filter": ["lex.uz"],
                "return_citations": True,
                "max_tokens": PERPLEXITY_ANSWER_MAX_TOKENS,
            },
            timeout=60,
        )
        
        if resp.status_code != 200:
            print(f"[PERPLEXITY] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        answer_text = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        
        # Find all citation numbers used in the text [1], [2], [6], etc.
        used_citation_nums = list(dict.fromkeys(re.findall(r'\[(\d+)\]', answer_text)))
        
        # Build sources dict - map citation numbers to URLs
        sources = {}
        for citation_num in used_citation_nums:
            idx = int(citation_num) - 1  # Convert to 0-indexed
            if 0 <= idx < len(citations):
                url = citations[idx]
                label = extract_domain_label(url)
                title = extract_title_from_url(url)
                sources[citation_num] = {
                    "id": int(citation_num),
                    "url": url,
                    "domain": urlparse(url).netloc if url else "",
                    "label": label,
                    "title": title,
                    "snippet": ""
                }
        
        # Also add any citations not referenced in text (for completeness)
        for i, url in enumerate(citations, 1):
            if str(i) not in sources:
                label = extract_domain_label(url)
                title = extract_title_from_url(url)
                sources[str(i)] = {
                    "id": i,
                    "url": url,
                    "domain": urlparse(url).netloc if url else "",
                    "label": label,
                    "title": title,
                    "snippet": ""
                }
        
        # Parse answer into blocks
        blocks = parse_answer_to_blocks_v2(answer_text, sources)
        
        # Generate context-aware related questions (don't rely on Perplexity's)
        related_questions = generate_related_questions(question, answer_text)
        
        return {
            "blocks": blocks,
            "sources": sources,
            "relatedQuestions": related_questions
        }
        
    except Exception as e:
        print(f"[PERPLEXITY] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def ask_perplexity(question: str) -> str:
    """Use Perplexity API - searches and answers in one call (legacy string response)."""
    
    if not PERPLEXITY_API_KEY:
        return None
    
    # Add context to focus on Uzbek law + multi-search strategy
    full_question = build_perplexity_multisearch_question(question)
    
    system_prompt = """Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

QOIDALAR:
1. Javoblarni o'zbek tilida, aniq va qisqa bering
2. Manbalardan foydalansangiz [1], [2], [3] kabi raqamlar bilan ko'rsating
3. Javob oxirida ALBATTA quyidagi formatda yozing:

Manbalar:
(Barcha foydalanilgan manbalarni [N] Nom - URL formatida yozing)

Tegishli savollar:
(5-6 ta tegishli savol yozing)

4. HECH QACHON ** belgisini ishlatmang - bold/qalin matn kerak emas"""
    
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_question}
                ],
                "search_domain_filter": ["lex.uz"],
                "return_citations": True,
                "max_tokens": PERPLEXITY_ANSWER_MAX_TOKENS,
            },
            timeout=30,
        )
        
        if resp.status_code != 200:
            print(f"[PERPLEXITY] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        
        # Add citations if not already in answer
        citations = data.get("citations", [])
        if citations and "Manbalar:" not in answer:
            answer += "\n\nManbalar:\n"
            for i, url in enumerate(citations, 1):
                title = extract_title_from_url(url)
                answer += f"[{i}] {title} - {url}\n"
        
        # Add related questions if not present
        if "Tegishli savollar:" not in answer:
            answer += "\n\nTegishli savollar:\n"
            answer += "- Bu mavzu bo'yicha boshqa qonunlar bormi?\n"
            answer += "- Qanday jarimalar ko'zda tutilgan?\n"
            answer += "- Bu qonun qachon kuchga kirgan?\n"
            answer += "- Istisnolar bormi?\n"
            answer += "- Amaliyotda qanday qo'llaniladi?\n"
        
        return answer
        
    except Exception as e:
        print(f"[PERPLEXITY] Failed: {e}")
        return None


def search_ask_with_provider(question: str, history: list = None) -> tuple[str, str]:
    """Main function with provider metadata.
    Priority: 1=DuckDuckGo (web), 2=Gemini, 3=Perplexity.
    """
    print(f"\n{'='*60}")
    print(f"SAVOL: {question}")
    if history:
        print(f"HISTORY: {len(history)} messages")
    print('='*60)
    
    # 1. DUCKDUCKGO (priority 1) - Top web results
    print("\n[WEB] Searching top web results (DuckDuckGo)...")
    web_results = search_web_top_results(question, max_results=WEB_MAX_RESULTS)

    # 1b. Multiple targeted queries (like Google AI): Perplexity suggests 4–5 angles; we run each and merge
    if PERPLEXITY_API_KEY:
        print("[WEB] Generating multiple targeted search queries (Perplexity)...")
        extra_queries = rewrite_queries_with_perplexity(question, max_queries=5)
        if extra_queries:
            if WEB_SEARCH_RETRY_DELAY > 0:
                time.sleep(WEB_SEARCH_RETRY_DELAY)
            all_results = list(web_results) if web_results else []
            for i, q in enumerate(extra_queries):
                print(f"[WEB] Targeted search {i+1}/{len(extra_queries)}: {q[:60]}{'...' if len(q) > 60 else ''}")
                part = search_web_top_results(q, max_results=WEB_MAX_RESULTS)
                if part:
                    all_results = _merge_web_results(all_results, part, prefer_uz=True)
                if i < len(extra_queries) - 1 and WEB_SEARCH_RETRY_DELAY > 0:
                    time.sleep(WEB_SEARCH_RETRY_DELAY)
            if all_results:
                web_results = all_results
                print(f"[WEB] Merged {len(web_results)} results (Uzbek/lex.uz preferred).")

    if web_results:
        gemini_summary = summarize_web_results_with_gemini(question, web_results)
        if gemini_summary:
            return gemini_summary, "web+gemini"
        return build_web_results_fallback_answer(web_results), "web-search"

    print("[WEB] No results, trying Gemini (priority 2)...")

    # 2. GEMINI (priority 2) - Google AI with Search Grounding
    if GEMINI_API_KEY:
        print("\n[GEMINI] Using Google Gemini with Search Grounding...")
        answer = ask_gemini_grounded(question, history=history)
        if answer:
            return answer, "gemini"
        print("[GEMINI] Failed, trying Perplexity...")
    
    # 3. PERPLEXITY (priority 3) - Final fallback
    if PERPLEXITY_API_KEY:
        print("\n[PERPLEXITY] Using Perplexity API...")
        answer = ask_perplexity(question)
        if answer:
            return answer, "perplexity"
        print("[PERPLEXITY] Failed")

    return "Javob topilmadi. Web/Gemini/Perplexity manbalaridan javob olib bo'lmadi.", "none"


def search_ask(question: str, history: list = None) -> str:
    """Main function: priority 1=DuckDuckGo, 2=Gemini, 3=Perplexity."""
    answer, _provider = search_ask_with_provider(question, history=history)
    return answer


def main():
    if len(sys.argv) < 2:
        print("Usage: python search_ask.py \"your question here\"")
        print("\nExamples:")
        print('  python search_ask.py "QQS stavkasi necha foiz?"')
        print('  python search_ask.py "Mehnat kodeksida yillik ta\'til muddati"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    answer = search_ask(question)
    
    print("\n" + "="*60)
    print("JAVOB:")
    print("="*60)
    print(answer)
    print("\n⚠️  Bu huquqiy maslahat emas.")


if __name__ == "__main__":
    main()
