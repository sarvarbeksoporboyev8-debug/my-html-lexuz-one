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
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
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

MUHIM: Javobni paragraflar bilan yoz. Har bir paragrafdan keyin qaysi manbadan olganingni [1], [2] kabi raqam bilan ko'rsat.
O'zbek tilida javob ber. Qisqa va aniq (2-4 paragraf).
Oxirida 5 ta tegishli savol yoz."""
    
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
        
        # Extract related questions
        related_questions = extract_related_questions(answer_text)
        
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
    
    # Default questions if none found
    if not questions:
        questions = [
            "Bu mavzu bo'yicha boshqa qonunlar bormi?",
            "Qanday jarimalar ko'zda tutilgan?",
            "Bu qonun qachon kuchga kirgan?",
            "Istisnolar bormi?",
            "Amaliyotda qanday qo'llaniladi?"
        ]
    
    return questions


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
    
    # Add sources
    sources = result.get("sources", [])
    if sources:
        answer += "\n\nManbalar:\n"
        for s in sources:
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
    
    full_question = f"O'zbekiston qonunchiligi bo'yicha savol (lex.uz saytidan ma'lumot): {question}"
    
    system_prompt = """Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

QOIDALAR:
1. Javoblarni o'zbek tilida, aniq va qisqa bering (2-4 paragraf)
2. Har bir paragrafda manbalarni [1], [2] kabi raqamlar bilan ko'rsating
3. Javob oxirida 5-6 ta tegishli savol yozing
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
                "search_recency_filter": "year",
            },
            timeout=60,
        )
        
        if resp.status_code != 200:
            print(f"[PERPLEXITY] Error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        data = resp.json()
        answer_text = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        
        # Build sources dict from citations
        sources = {}
        for i, url in enumerate(citations, 1):
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
        
        # Extract related questions
        related_questions = extract_related_questions(answer_text)
        
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
    """Main function: Perplexity first (best), then Gemini, then Local FTS."""
    
    print(f"\n{'='*60}")
    print(f"SAVOL: {question}")
    if history:
        print(f"HISTORY: {len(history)} messages")
    print('='*60)
    
    # 1. PERPLEXITY - Best quality
    if PERPLEXITY_API_KEY:
        print("\n[PERPLEXITY] Using Perplexity API...")
        answer = ask_perplexity(question)
        if answer:
            return answer
        print("[PERPLEXITY] Failed, trying Gemini...")
    
    # 2. GEMINI - Google AI Mode with Search Grounding
    if GEMINI_API_KEY:
        print("\n[GEMINI] Using Google Gemini with Search Grounding...")
        answer = ask_gemini_grounded(question, history=history)
        if answer:
            return answer
        print("[GEMINI] Failed, trying Local FTS...")

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
