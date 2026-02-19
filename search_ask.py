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
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup

# Perplexity API (primary)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# Google Gemini API with Search Grounding (fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")



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


def search_web_top_results(query: str, max_results: int = 8) -> list:
    """Search web results using DuckDuckGo HTML endpoint (no API key required)."""
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            },
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"[WEB] Search failed: {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        seen_urls = set()

        for block in soup.select(".result"):
            link_el = block.select_one("a.result__a")
            if not link_el:
                continue

            url = link_el.get("href", "").strip()
            if not url:
                continue

            if "uddg=" in url:
                try:
                    parsed = parse_qs(urlparse(url).query)
                    url = unquote(parsed.get("uddg", [url])[0])
                except Exception:
                    pass

            title = link_el.get_text(" ", strip=True)
            snippet_el = block.select_one(".result__snippet")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            results.append({
                "id": len(results) + 1,
                "title": title[:180] if title else extract_title_from_url(url),
                "url": url,
                "snippet": snippet[:400],
                "label": extract_domain_label(url),
            })

            if len(results) >= max_results:
                break

        return results

    except Exception as e:
        print(f"[WEB] Search exception: {e}")
        return []


def summarize_web_results_with_gemini(question: str, results: list) -> str:
    """Summarize fetched web results with Gemini, citing [1], [2], ... markers."""
    if not GEMINI_API_KEY or not results:
        return None

    context_lines = []
    for item in results:
        context_lines.append(
            f"[{item['id']}] {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('snippet', '')}"
        )

    prompt = f"""Siz O'zbekiston huquqiy masalalari bo'yicha yordamchi AI siz.

SAVOL:
{question}

INTERNET MANBALARI (top natijalar):
{chr(10).join(context_lines)}

QOIDALAR:
1. Faqat berilgan manbalarga tayangan holda javob bering
2. O'zbek tilida yozing, 2-4 paragraf, aniq va qisqa
3. Har paragraf oxirida [1], [2] kabi iqtibos raqamlarini qo'ying
4. Taxmin qilmang; manbada bo'lmasa ochiq ayting
5. Oxirida quyidagi bo'limlarni yozing:

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
                        "maxOutputTokens": 1200,
                    },
                },
                timeout=30,
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
        "- Maktab hududida reklama bo'yicha alohida cheklovlar bormi?",
        "- Kripto xizmatlar logotipidan foydalanish uchun ruxsat talab qilinadimi?",
        "- Voyaga yetmaganlarga qaratilgan reklama bo'yicha talablar qanday?",
        "- Qaysi davlat organi bu masalani nazorat qiladi?",
    ]
    return "\n".join(lines)


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
                "search_recency_filter": "year",
                "max_tokens": 4000,
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
                "max_tokens": 4000,
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
    """Main function with provider metadata: web-search first, then Gemini, then Perplexity."""
    
    print(f"\n{'='*60}")
    print(f"SAVOL: {question}")
    if history:
        print(f"HISTORY: {len(history)} messages")
    print('='*60)
    
    # 1. WEB SEARCH - Top internet results first (5-10)
    print("\n[WEB] Searching top web results...")
    web_results = search_web_top_results(question, max_results=10)
    if web_results:
        gemini_summary = summarize_web_results_with_gemini(question, web_results)
        if gemini_summary:
            return gemini_summary, "web+gemini"
        return build_web_results_fallback_answer(web_results), "web-search"
    print("[WEB] No results found, trying Gemini...")

    # 2. GEMINI - Google AI Mode with Search Grounding
    if GEMINI_API_KEY:
        print("\n[GEMINI] Using Google Gemini with Search Grounding...")
        answer = ask_gemini_grounded(question, history=history)
        if answer:
            return answer, "gemini"
        print("[GEMINI] Failed, trying Perplexity...")
    
    # 3. PERPLEXITY - Final provider fallback
    if PERPLEXITY_API_KEY:
        print("\n[PERPLEXITY] Using Perplexity API...")
        answer = ask_perplexity(question)
        if answer:
            return answer, "perplexity"
        print("[PERPLEXITY] Failed")

    return "Javob topilmadi. Web/Gemini/Perplexity manbalaridan javob olib bo'lmadi.", "none"


def search_ask(question: str, history: list = None) -> str:
    """Main function: web search first, then Gemini, then Perplexity."""
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
