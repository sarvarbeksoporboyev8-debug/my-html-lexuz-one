#!/usr/bin/env python3
"""
Search-based legal Q&A using SearXNG + Scraping + DeepSeek
Cost: ~$0.001 per query

Usage:
    python search_ask.py "QQS stavkasi necha foiz?"
    python search_ask.py "Mehnat kodeksida yillik ta'til muddati"
"""

import os
import sys
import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: local ultra-fast search (SQLite FTS5) over your downloaded lex.uz HTML pages.
# Build the DB once with:
#   python local_fts_index.py --html-dir ./lex_html --db ./data/lexuz_fts.sqlite3
# Then set:
#   export LEXUZ_LOCAL_FTS_DB=./data/lexuz_fts.sqlite3
# and this script will use local search instead of DuckDuckGo.

LEXUZ_LOCAL_FTS_DB = os.getenv("LEXUZ_LOCAL_FTS_DB", "").strip()
if LEXUZ_LOCAL_FTS_DB:
    try:
        from local_fts_search import LexUZFTSSearcher  # type: ignore
    except Exception:
        LexUZFTSSearcher = None

# Config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Perplexity API
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# Google Gemini API with Search Grounding (like Google AI Mode!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Bright Data proxy for scraping
BRIGHTDATA_HOST = os.getenv("BRIGHTDATA_HOST", "brd.superproxy.io")
BRIGHTDATA_PORT = os.getenv("BRIGHTDATA_PORT", "33335")
BRIGHTDATA_USERNAME = os.getenv("BRIGHTDATA_USERNAME", "")
BRIGHTDATA_PASSWORD = os.getenv("BRIGHTDATA_PASSWORD", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def get_proxy():
    """Get Bright Data proxy config."""
    if BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD:
        proxy_url = f"http://{BRIGHTDATA_USERNAME}:{BRIGHTDATA_PASSWORD}@{BRIGHTDATA_HOST}:{BRIGHTDATA_PORT}"
        return {"http": proxy_url, "https": proxy_url}
    return None


def search_duckduckgo(query: str, num_results: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML (no API needed, no blocks)."""
    
    search_query = f"site:lex.uz {query}"
    
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        
        if resp.status_code != 200:
            print(f"[SEARCH] DuckDuckGo returned {resp.status_code}")
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        results = []
        for r in soup.select(".result"):
            link = r.select_one("a.result__a")
            snippet_el = r.select_one(".result__snippet")
            
            if link:
                href = link.get("href", "")
                title = link.get_text(strip=True)
                
                # DuckDuckGo wraps URLs, extract actual URL
                if "uddg=" in href:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]
                
                if "lex.uz" in href:
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    })
        
        print(f"[SEARCH] Found {len(results)} results from DuckDuckGo")
        return results[:num_results]
        
    except Exception as e:
        print(f"[SEARCH] DuckDuckGo search failed: {e}")
        return []





def scrape_page(url: str, max_chars: int = 4000) -> str:
    """Scrape text content from a URL."""
    try:
        proxy = get_proxy()
        resp = requests.get(url, headers=HEADERS, proxies=proxy, timeout=15)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Get main content
        main = soup.select_one("article, .content, .document-content, main, #content")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        
        # Clean up
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        
        return text[:max_chars]
    except Exception as e:
        print(f"[SCRAPE] Failed {url}: {e}")
        return ""


def scrape_all(urls: list[str], max_chars_per_page: int = 4000) -> list[dict]:
    """Scrape multiple URLs in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_page, url, max_chars_per_page): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    results.append({"url": url, "content": text})
                    print(f"[SCRAPE] Got {len(text)} chars from {url[:50]}...")
            except Exception as e:
                print(f"[SCRAPE] Error {url}: {e}")
    return results


def ask_deepseek(question: str, contexts: list[dict]) -> str:
    """Send question + contexts to DeepSeek."""
    
    # Build context string
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(f"[SOURCE {i}] {ctx['url']}\n{ctx['content'][:3000]}")
    
    context_str = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Siz O'zbekiston qonunchiligini yaxshi biladigan huquqiy yordamchi AI siz.

Quyidagi manbalar va o'zingizning bilimlaringiz asosida savolga aniq javob bering.
Manbalardan foydalansangiz, [SOURCE N] formatida ko'rsating.
Javob qisqa va aniq bo'lsin.

MANBALAR (lex.uz dan):
{context_str}

SAVOL: {question}

JAVOB (o'zbek tilida):"""

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1500,
            },
            timeout=60,
        )
        
        if resp.status_code != 200:
            return f"DeepSeek API error: {resp.status_code} - {resp.text[:200]}"
        
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        
        # Add sources
        answer += "\n\n**Manbalar:**\n"
        for i, ctx in enumerate(contexts, 1):
            answer += f"{i}. {ctx['url']}\n"
        
        return answer
        
    except Exception as e:
        return f"DeepSeek error: {e}"


def ask_gemini_grounded(question: str) -> str:
    """Use Google Gemini with Search Grounding - THIS IS GOOGLE AI MODE!"""
    
    if not GEMINI_API_KEY:
        return None
    
    full_question = f"O'zbekiston qonunchiligi haqida savol (lex.uz saytidan ma'lumot izla): {question}"
    
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [
                    {
                        "parts": [{"text": full_question}]
                    }
                ],
                "tools": [
                    {"google_search": {}}
                ]
            },
            timeout=30,
        )
        
        if resp.status_code != 200:
            print(f"[GEMINI] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        
        # Extract answer
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract citations from grounding metadata
        grounding = data["candidates"][0].get("groundingMetadata", {})
        chunks = grounding.get("groundingChunks", [])
        
        if chunks:
            answer += "\n\n**Manbalar:**\n"
            seen_urls = set()
            for i, chunk in enumerate(chunks, 1):
                url = chunk.get("web", {}).get("uri", "")
                title = chunk.get("web", {}).get("title", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    answer += f"{i}. {url}\n"
        
        return answer
        
    except Exception as e:
        print(f"[GEMINI] Failed: {e}")
        return None


def ask_perplexity(question: str) -> str:
    """Use Perplexity API - searches and answers in one call (like Google AI Mode)."""
    
    if not PERPLEXITY_API_KEY:
        return None
    
    # Add context to focus on Uzbek law
    full_question = f"O'zbekiston qonunchiligi bo'yicha savol (lex.uz saytidan ma'lumot): {question}"
    
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",  # or "sonar-pro" for better quality
                "messages": [
                    {
                        "role": "system",
                        "content": "Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz. Javoblarni o'zbek tilida, aniq va qisqa bering. Manbalarni ko'rsating."
                    },
                    {
                        "role": "user", 
                        "content": full_question
                    }
                ],
                "search_domain_filter": ["lex.uz"],  # Focus on lex.uz
                "return_citations": True,
                "search_recency_filter": "year",
            },
            timeout=30,
        )
        
        if resp.status_code != 200:
            print(f"[PERPLEXITY] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        
        # Add citations if available
        citations = data.get("citations", [])
        if citations:
            answer += "\n\n**Manbalar:**\n"
            for i, url in enumerate(citations, 1):
                answer += f"{i}. {url}\n"
        
        return answer
        
    except Exception as e:
        print(f"[PERPLEXITY] Failed: {e}")
        return None


def search_ask(question: str) -> str:
    """Main function: Local FTS first (cheapest), then Perplexity/Gemini, then DuckDuckGo."""
    
    print(f"\n{'='*60}")
    print(f"SAVOL: {question}")
    print('='*60)
    
    # 1. LOCAL FTS - Cheapest (~$0.001/query with DeepSeek)
    if LEXUZ_LOCAL_FTS_DB and LexUZFTSSearcher is not None:
        try:
            print("\n[LOCAL] Searching local LexUZ index (SQLite FTS5)...")
            searcher = LexUZFTSSearcher(Path(LEXUZ_LOCAL_FTS_DB))
            hits = searcher.search(question, k=8, per_doc=2, candidate_k=80, with_text=True)

            if hits:
                # Convert hits -> contexts for DeepSeek.
                contexts = []
                for h in hits:
                    contexts.append({"url": h.url, "content": (h.text or "")[:4000]})

                print(f"\n[LOCAL] Found {len(hits)} chunks, generating answer with DeepSeek...")
                return ask_deepseek(question, contexts)
            else:
                print("[LOCAL] No matches found, trying web search...")
        except Exception as e:
            print(f"[LOCAL] Failed ({e}), trying web search...")
    
    # 2. GEMINI - Google AI Mode (~$0.0035/query)
    if GEMINI_API_KEY:
        print("\n[GEMINI] Using Google Gemini with Search Grounding...")
        answer = ask_gemini_grounded(question)
        if answer:
            return answer
        print("[GEMINI] Failed, trying Perplexity...")
    
    # 3. PERPLEXITY (~$0.006/query)
    if PERPLEXITY_API_KEY:
        print("\n[PERPLEXITY] Using Perplexity API...")
        answer = ask_perplexity(question)
        if answer:
            return answer
        print("[PERPLEXITY] Failed, falling back to DuckDuckGo + DeepSeek...")

    # Fallback: DuckDuckGo + scrape + DeepSeek (~$0.001/query)
    # 1. Search
    print("\n[1/3] Searching...")
    results = search_duckduckgo(question, num_results=5)

    if not results:
        return "Qidiruv natijalari topilmadi. Internetga ulanishni tekshiring."

    # 2. Scrape
    print("\n[2/3] Scraping pages...")
    urls = [r["url"] for r in results]
    contexts = scrape_all(urls)

    if not contexts:
        return "Sahifalarni o'qib bo'lmadi."

    # 3. Ask LLM
    print("\n[3/3] Generating answer...")
    return ask_deepseek(question, contexts)


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
