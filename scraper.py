import json
import os
import re
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

DEFAULT_URLS = [
    "https://manavrachnaonline.com/"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "what", "tell",
    "me", "about", "of", "and", "or", "if", "i", "you", "your", "my", "we"
}

def fetch_page(url, timeout=15):
    session = requests.Session()
    session.max_redirects = 50
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_page_content(url, html):
    soup = BeautifulSoup(html, "lxml")
    records = []
    links = set()
    base_domain = "manavrachnaonline.com"

    # Extract Content
    title = clean_text(soup.title.string if soup.title else "")
    if title:
        records.append({"source": url, "type": "title", "text": title})

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        text = clean_text(tag.get_text(separator=" "))
        if text:
            records.append({"source": url, "type": tag.name, "text": text})

    for paragraph in soup.find_all("p"):
        text = clean_text(paragraph.get_text(separator=" "))
        if text:
            records.append({"source": url, "type": "paragraph", "text": text})

    for li in soup.find_all("li"):
        text = clean_text(li.get_text(separator=" "))
        if text:
            records.append({"source": url, "type": "list", "text": text})

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean_text(cell.get_text(separator=" ")) for cell in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join([c for c in cells if c]))
        if rows:
            records.append({"source": url, "type": "table", "text": " \n ".join(rows)})

    if not records:
        body = clean_text(soup.get_text(separator=" "))
        if body:
            records.append({"source": url, "type": "body", "text": body})

    for record in records:
        record["text"] = re.sub(r"\s{2,}", " ", record["text"]).strip()

    # Extract Links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = f"https://{base_domain}{href}"
        
        if base_domain in href and "tel:" not in href and "mailto:" not in href:
            clean_href = href.split("#")[0].split("?")[0].rstrip("/")
            if clean_href.startswith("http"):
                links.add(clean_href)

    return records, links


def scrape_urls(urls=None, max_pages=100, max_workers=10):
    target_urls = set(urls or DEFAULT_URLS)
    all_records = []
    scraped_urls = set()
    queue = list(target_urls)
    
    lock = threading.Lock()
    
    def process_url(url):
        nonlocal all_records
        try:
            print(f"Scraping: {url}")
            html = fetch_page(url)
            
            with lock:
                scraped_urls.add(url)
            
            # Extract content and links
            extracted, new_links = scrape_page_content(url, html)
            with lock:
                all_records.extend(extracted)
                for link in new_links:
                    if link not in scraped_urls and link not in queue:
                        if not max_pages or (len(scraped_urls) + len(queue)) < max_pages:
                            queue.append(link)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while queue:
            if max_pages and len(scraped_urls) >= max_pages:
                break
                
            # Take a batch of URLs from the queue
            with lock:
                current_batch = []
                while queue and (not max_pages or len(scraped_urls) + len(current_batch) < max_pages):
                    u = queue.pop(0)
                    if u not in scraped_urls:
                        current_batch.append(u)
                    if len(current_batch) >= max_workers:
                        break
            
            if not current_batch:
                break
                
            executor.map(process_url, current_batch)
            
    return all_records


def save_scraped_data(data, path):
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pages": data,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_scraped_data(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
            return payload.get("pages", [])
    except Exception:
        return []


def update_scraped_data(path, max_pages=100):
    data = scrape_urls(max_pages=max_pages)
    save_scraped_data(data, path)
    return data


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b[\w']+\b", text)
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def search_content(question, pages, max_results=8):
    question_lower = question.lower()
    question_tokens = tokenize(question)
    
    if not question_tokens:
        orig_tokens = [t for t in re.findall(r"\b[\w']+\b", question.lower()) if len(t) > 1]
        if not orig_tokens:
            return "I did not understand that. Please ask about admissions, courses, faculty, contact, or other college details."
        question_tokens = orig_tokens

    weights = {
        "title": 10,
        "h1": 8,
        "h2": 6,
        "h3": 4,
        "h4": 3,
        "h5": 2,
        "paragraph": 1.5,
        "list": 1.2,
        "table": 2,
        "body": 0.5
    }

    best_matches = []
    for page in pages:
        page_text = page.get("text", "")
        page_type = page.get("type", "body")
        page_text_lower = page_text.lower()
        
        score = 0
        # Token matching
        for token in question_tokens:
            if token in page_text_lower:
                score += 1
        
        # Phrase matching bonus
        if question_lower in page_text_lower:
            score += 10
        elif len(question_tokens) > 1:
            # Check for partial phrases (pairs of tokens)
            for i in range(len(question_tokens) - 1):
                phrase = f"{question_tokens[i]} {question_tokens[i+1]}"
                if phrase in page_text_lower:
                    score += 5
        
        if score > 0:
            weight = weights.get(page_type, 1)
            final_score = score * weight
            best_matches.append((final_score, page_text, page.get("source", ""), page_type))

    if not best_matches:
        return "I'm sorry, I couldn't find specific information about that on the Manav Rachna website. You might want to check the official site directly at https://manavrachna.edu.in/."

    # Sort by score and then by length
    best_matches.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    
    answer_parts = []
    seen_text = set()
    for score, text, source, kind in best_matches:
        # Avoid duplicate or near-duplicate content
        text_normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if text_normalized not in seen_text:
            answer_parts.append(f"[{kind.upper()}] {text}\n(Source: {source})")
            seen_text.add(text_normalized)
        if len(answer_parts) >= max_results:
            break

    return "\n\n".join(answer_parts)


def get_gemini_response(api_key, question, context_data):
    if not api_key:
        return "Gemini API key is not configured."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    You are a helpful assistant for Manav Rachna College. 
    Use the following scraped data from the college website to answer the student's question.
    If the data doesn't contain the answer, politely say you don't know and suggest visiting the official website.
    
    Scraped Data Context:
    {context_data}
    
    Student Question: {question}
    
    Answer (USE ONLY HTML TAGS like <p>, <ul>, <li>, <strong> for formatting. DO NOT USE markdown symbols like * or **. Ensure the response is valid HTML content):
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Remove markdown code blocks if present
        text = re.sub(r"```(?:html)?", "", text)
        # Remove any markdown bold/italic markers
        text = text.replace("**", "").replace("*", "")
        return text.strip()
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"


if __name__ == "__main__":
    print("Starting scraper...")
    update_scraped_data("scraped_data.json", max_pages=100)
    print("Scraping completed. Data saved to scraped_data.json.")
