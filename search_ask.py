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

# Config - LLM provider (OpenRouter or DeepSeek)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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


def tidy_question(question: str) -> str:
    """Use Gemma (cheap) to clean up messy questions."""
    
    if not OPENROUTER_API_KEY:
        return question  # No Gemma, return as-is
    
    prompt = f"""Siz savol tozalovchi yordamchisiz. Foydalanuvchi savolini to'g'ri o'zbek tilida qayta yozing.

VAZIFA:
- Imlo xatolarini tuzating
- Grammatikani to'g'rilang  
- Qisqartmalarni to'liq yozing (QQS = Qo'shilgan qiymat solig'i)
- Savol tushunarli va aniq bo'lsin
- FAQAT savolni qaytaring, boshqa hech narsa yozmang

MISOL:
Kirish: "qqsni nechchi foiz toliman kerak bilasizmi"
Chiqish: "Qo'shilgan qiymat solig'i (QQS) necha foiz to'lanadi?"

MISOL:
Kirish: "mashina olsam qancha soliq toliman"
Chiqish: "Avtomobil sotib olsam qancha soliq to'lashim kerak?"

Kirish: {question}
Chiqish:"""

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
            },
            timeout=30,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            tidied = data["choices"][0]["message"]["content"].strip()
            if tidied:
                return tidied
    except Exception:
        pass
    
    return question  # Fallback to original


def ask_llm(question: str, contexts: list[dict]) -> str:
    """Send question + contexts to DeepSeek (accurate for legal answers)."""
    
    if not DEEPSEEK_API_KEY:
        return "DEEPSEEK_API_KEY topilmadi."
    
    # Build context string with source numbers
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(f"[{i}] {ctx['url']}\n{ctx['content'][:3000]}")
    
    context_str = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Siz O'zbekiston qonunchiligi bo'yicha yordamchisiz.

SAVOL: {question}

QOIDALAR:
1. FAQAT so'ralgan savolga javob bering - ortiqcha ma'lumot bermang
2. Javob 2-4 jumla bo'lsin, kerak bo'lsa ro'yxat qo'shing
3. Manbalardan foydalansangiz [1], [2] kabi raqam qo'ying
4. Oxirida quyidagi formatda yozing:

Manbalar:
[1] Nom - URL
[2] Nom - URL

Tegishli savollar:
- Savol?
- Savol?
- Savol?
- Savol?
- Savol?

MANBALAR:
{context_str}

JAVOB (qisqa va aniq):"""

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
                "max_tokens": 2000,
            },
            timeout=60,
        )
        
        if resp.status_code != 200:
            return f"DeepSeek API error: {resp.status_code} - {resp.text[:200]}"
        
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        
        # Check if sources section exists, if not add it
        if "Manbalar:" not in answer:
            answer += "\n\nManbalar:\n"
            for i, ctx in enumerate(contexts, 1):
                # Extract title from URL or use generic
                title = extract_title_from_url(ctx['url'])
                answer += f"[{i}] {title} - {ctx['url']}\n"
        
        # Check if related questions exist, if not add generic ones
        if "Tegishli savollar:" not in answer:
            answer += "\n\nTegishli savollar:\n"
            answer += "- Bu mavzu bo'yicha boshqa qonunlar bormi?\n"
            answer += "- Qanday jarimalar ko'zda tutilgan?\n"
            answer += "- Bu qonun qachon kuchga kirgan?\n"
            answer += "- Istisnolar bormi?\n"
            answer += "- Amaliyotda qanday qo'llaniladi?\n"
        
        return answer
        
    except Exception as e:
        return f"DeepSeek error: {e}"


def extract_title_from_url(url: str) -> str:
    """Extract a readable title from lex.uz URL."""
    # Try to get doc ID from URL
    import re
    match = re.search(r'/docs/(-?\d+)', url)
    if match:
        doc_id = match.group(1)
        return f"Lex.uz hujjat #{doc_id}"
    return "Lex.uz hujjat"


def ask_gemini_grounded(question: str, history: list = None) -> str:
    """Use Google Gemini with Search Grounding - THIS IS GOOGLE AI MODE!"""
    
    if not GEMINI_API_KEY:
        return None
    
    # Build conversation contents
    contents = []
    
    # Add chat history if provided
    if history:
        for msg in history[-5:]:  # Last 5 messages max
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("content", "")
            if text:
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })
    
    # Add current question with instructions
    full_question = f"""{question}

O'zbek tilida javob ber. Qisqa va aniq (2-5 jumla). Manbalarga link ber. 5 ta tegishli savol ham ber."""
    
    contents.append({
        "role": "user",
        "parts": [{"text": full_question}]
    })
    
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
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
        
        if resp.status_code != 200:
            print(f"[GEMINI] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        
        # Extract answer
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract citations from grounding metadata if not in answer
        grounding = data["candidates"][0].get("groundingMetadata", {})
        chunks = grounding.get("groundingChunks", [])
        
        if chunks and "Manbalar:" not in answer:
            answer += "\n\nManbalar:\n"
            seen_urls = set()
            i = 1
            for chunk in chunks:
                url = chunk.get("web", {}).get("uri", "")
                title = chunk.get("web", {}).get("title", "") or extract_title_from_url(url)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    answer += f"[{i}] {title} - {url}\n"
                    i += 1
        
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
        print(f"[GEMINI] Failed: {e}")
        return None


def ask_perplexity(question: str) -> str:
    """Use Perplexity API - searches and answers in one call (like Google AI Mode)."""
    
    if not PERPLEXITY_API_KEY:
        return None
    
    # Add context to focus on Uzbek law
    full_question = f"O'zbekiston qonunchiligi bo'yicha savol (lex.uz saytidan ma'lumot): {question}"
    
    system_prompt = """Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

QOIDALAR:
1. Javoblarni o'zbek tilida, aniq va qisqa bering
2. Manbalardan foydalansangiz [1], [2], [3] kabi raqamlar bilan ko'rsating
3. Javob oxirida ALBATTA quyidagi formatda yozing:

Manbalar:
(Barcha foydalanilgan manbalarni [N] Nom - URL formatida yozing)

Tegishli savollar:
(5-6 ta tegishli savol yozing)"""
    
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
                "search_recency_filter": "year",
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


def search_ask(question: str, history: list = None) -> str:
    """Main function: Gemini first (best), then Perplexity, then DuckDuckGo + DeepSeek."""
    
    print(f"\n{'='*60}")
    print(f"SAVOL: {question}")
    if history:
        print(f"HISTORY: {len(history)} messages")
    print('='*60)
    
    # 1. GEMINI - Primary choice (cheap + has web search)
    if GEMINI_API_KEY:
        print("\n[GEMINI] Using Google Gemini with Search Grounding...")
        answer = ask_gemini_grounded(question, history=history)
        if answer:
            return answer
        print("[GEMINI] Failed, trying Perplexity...")
    
    # 2. PERPLEXITY - Backup
    if PERPLEXITY_API_KEY:
        print("\n[PERPLEXITY] Using Perplexity API...")
        answer = ask_perplexity(question)
        if answer:
            return answer
        print("[PERPLEXITY] Failed, falling back to DuckDuckGo + DeepSeek...")

    # 3. LOCAL FTS + DeepSeek - Fallback
    if LEXUZ_LOCAL_FTS_DB and LexUZFTSSearcher is not None:
        try:
            print("\n[LOCAL] Searching local LexUZ index...")
            searcher = LexUZFTSSearcher(Path(LEXUZ_LOCAL_FTS_DB))
            hits = searcher.search(question, k=8, per_doc=2, candidate_k=80, with_text=True)

            if hits:
                contexts = []
                for h in hits:
                    contexts.append({"url": h.url, "content": (h.text or "")[:4000]})

                print(f"\n[LOCAL] Found {len(hits)} chunks, generating answer with DeepSeek...")
                return ask_llm(question, contexts)
        except Exception as e:
            print(f"[LOCAL] Failed ({e})")

    # 4. DuckDuckGo + scrape + DeepSeek - Last resort
    print("\n[1/3] Searching DuckDuckGo...")
    results = search_duckduckgo(question, num_results=5)

    if not results:
        return "Qidiruv natijalari topilmadi. Internetga ulanishni tekshiring."

    print("\n[2/3] Scraping pages...")
    urls = [r["url"] for r in results]
    contexts = scrape_all(urls)

    if not contexts:
        return "Sahifalarni o'qib bo'lmadi."

    print("\n[3/3] Generating answer...")
    return ask_llm(question, contexts)


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
