import sys
import os
import re
import json
from datetime import datetime
import logging
from typing import List, Tuple
import requests
from urllib.parse import urlparse
import socket
import time

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: The 'google-generativeai' package is not installed.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)

# 📜 Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# 🔐 API keys
GEMINI_API_KEY = "AIzaSyBFIwjEW0AGIgF4U_9nvYSCiBs4lhU2pl8"  # Your Gemini API key
SERPSTACK_API_KEY = "ca6d82e72794523fe0bc2faca15d9853"  # Replace with your Serpstack API key (get from https://serpstack.com/)

# 🗣 Arabic claim to fact-check
arabic_claim = " المواليين غلو علينا المعيشة في السومة تاع الخرفان  "

# 🧠 System prompt for Gemini API
system_prompt = (
    "أنت مساعد متخصص في التحقق من الأخبار باللغة العربية. "
    "عندما يقدم لك المستخدم ادعاء، تحقق من صحته باستخدام المصادر المتاحة على الإنترنت. "
    "قدم النتيجة بصيغة منظمة تحتوي على العناصر التالية:\n"
    "1. النتيجة: (صحيح/خاطئ/غير مؤكد)\n"
    "2. الشرح: تفسير مختصر لسبب النتيجة\n"
    "3. الروابط: روابط موثوقة من مواقع مثل مسبار، فتبينوا، رويترز، أو غيرها\n\n"
    "الرجاء أن يكون التنسيق واضحًا بحيث يمكن استخلاص كل جزء برمجيًا بسهولة."
)

# 🧾 User prompt for Gemini API
user_prompt = f"هل العبارة التالية صحيحة؟\n\n\"{arabic_claim}\"\n\nيرجى التحقق منها باستخدام مصادر موثوقة مثل مسبار، فتبينوا، أو وكالات الأنباء، وقدم تفاصيل دقيقة."

# 🌐 Trusted domains and fallback URLs
TRUSTED_DOMAINS = [
    "misbar.com",
    "fatabyyano.net",
    "reuters.com",
    "bbc.com",
    "aljazeera.net",
    "apnews.com",
    "france24.com",
    "aps.dz"
]
FALLBACK_SOURCES = [
    "https://www.reuters.com",
    "https://www.aps.dz",
    "https://www.bbc.com"
]

# Source name to URL mapping for mentions
SOURCE_MAPPING = {
    "رويترز": "https://www.reuters.com",
    "وكالة الأنباء الجزائرية": "https://www.aps.dz",
    "فرانس برس": "https://www.afp.com",
    "أسوشيتد برس": "https://www.apnews.com",
    "مسبار": "https://www.misbar.com",
    "فتبينوا": "https://fatabyyano.net"
}

def check_network() -> bool:
    """Check network connectivity by pinging google.com."""
    try:
        socket.create_connection(("google.com", 80), timeout=5)
        return True
    except OSError:
        logging.error("Network connectivity check failed. Ensure you have an active internet connection.")
        return False

def validate_inputs(gemini_key: str, serpstack_key: str, claim: str) -> bool:
    """Validate API keys and claim text."""
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        logging.error("Invalid or missing Gemini API key.")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        return False
    if not serpstack_key or serpstack_key == "YOUR_SERPSTACK_API_KEY_HERE":
        logging.warning("Invalid or missing Serpstack API key. Web search will use fallback sources.")
        print("Get your API key from: https://serpstack.com/ (optional for Gemini-only mode)")
    if not claim.strip():
        logging.error("Claim text is empty.")
        return False
    if not check_network():
        print("Please check your internet connection and try again.")
        return False
    return True

def call_gemini_api(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Call Gemini API to process the fact-checking request."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000
            }
        )
        if response.text:
            logging.info("Successfully received response from Gemini API.")
            logging.debug(f"Gemini response snippet: {response.text[:200]}...")
            return response.text
        logging.error("Empty response from Gemini API.")
        return ""
    except Exception as e:
        logging.error(f"Error calling Gemini API: {str(e)}")
        print("Possible issues:")
        print("- Invalid API key: Verify your key at https://aistudio.google.com/app/apikey")
        print("- Network error: Check your internet connection")
        print("- Rate limits: Ensure you haven't exceeded your API quota")
        return ""

def search_web_for_sources(claim: str, api_key: str) -> List[str]:
    """Search the web using Serpstack API and return trusted source links."""
    if not api_key or api_key == "YOUR_SERPSTACK_API_KEY_HERE":
        logging.warning("Serpstack API key missing or invalid. Using fallback sources.")
        return FALLBACK_SOURCES

    for attempt in range(3):  # Retry up to 3 times
        try:
            url = "http://api.serpstack.com/search"  # Serpstack endpoint
            params = {
                "access_key": api_key,
                "query": claim,
                "hl": "ar",  # Arabic language
                "gl": "dz",  # Algeria
                "num": 10
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            results = response.json()
            logging.debug(f"Serpstack response: {json.dumps(results, ensure_ascii=False)[:200]}...")
            organic_results = results.get("organic_results", [])
            links = []
            for result in organic_results:
                link = result.get("url", "")
                if link and any(domain in urlparse(link).netloc for domain in TRUSTED_DOMAINS):
                    links.append(link)
            links = list(dict.fromkeys(links))  # Deduplicate
            logging.info(f"Found {len(links)} trusted source links via Serpstack.")
            return links if links else FALLBACK_SOURCES
        except requests.exceptions.HTTPError as e:
            logging.error(f"Serpstack HTTP error (attempt {attempt + 1}/3): {str(e)}")
            if response.status_code == 401:
                print("Serpstack API key invalid. Get a valid key from: https://serpstack.com/")
                return FALLBACK_SOURCES    #return FALLBACK_SOURCES
            elif response.status_code == 429:
                print("Serpstack rate limit exceeded. Please try again later.")
                return FALLBACK_SOURCES
        except requests.exceptions.RequestException as e:
            logging.error(f"Serpstack request error (attempt {attempt + 1}/3): {str(e)}")
        if attempt < 2:
            logging.info(f"Retrying Serpstack request in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    print("Possible issues:")
    print("- Invalid API key: Verify your key at https://serpstack.com/")
    print("- Network error: Check your internet connection")
    print("- Serpstack server issues: Try again later or check https://www.reuters.com, https://www.aps.dz manually")
    return FALLBACK_SOURCES

def parse_response(response: str) -> Tuple[str, str, List[str]]:
    """Parse the Gemini API response to extract result, explanation, and links."""
    # Extract result
    result_match = re.search(r"(?:النتيجة|النتائج|النتيجه)\s*[:：]\s*\*?\*?\s*(صحيح|خاطئ|غير مؤكد)\s*\*?\*?", response, re.IGNORECASE | re.DOTALL)
    result = result_match.group(1) if result_match else "غير متوفر"
    
    # Extract explanation
    explanation_match = re.search(r"(?:الشرح|التفسير)\s*[:：]\s*(.*?)(?=(?:\n\s*(?:الروابط|3\.)|$))", response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else "غير متوفر"
    
    # Extract links
    links = re.findall(r'(https?://[^\s<>"\']+)', response)
    links = [link.rstrip('.,;') for link in links if any(domain in urlparse(link).netloc for domain in TRUSTED_DOMAINS)]
    
    # Extract source mentions if no links found
    if not links:
        for source_name, source_url in SOURCE_MAPPING.items():
            if source_name in response and source_url not in links:
                links.append(source_url)
    links = list(dict.fromkeys(links))  # Deduplicate
    
    logging.debug(f"Parsed result: {result}")
    logging.debug(f"Parsed explanation: {explanation[:100]}...")
    logging.debug(f"Parsed links: {links}")
    
    return result, explanation, links

def save_results(
    claim: str,
    result: str,
    explanation: str,
    gemini_links: List[str],
    search_links: List[str],
    model: str,
    file_path: str
) -> None:
    """Save fact-checking results to a JSON file."""
    try:
        # Deduplicate all links
        all_links = list(dict.fromkeys(gemini_links + search_links))
        fact_result = {
            "claim": claim,
            "result": result,
            "explanation": explanation,
            "sources": {
                "gemini_links": [link for link in gemini_links if link in all_links],
                "search_links": [link for link in search_links if link in all_links]
            },
            "metadata": {
                "model_used": model,
                "claim_length": len(claim),
                "response_length": len(explanation) + sum(len(link) for link in all_links),
                "total_links_count": len(all_links),
                "date_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(fact_result, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved successfully to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        print(f"Could not save results to {file_path}. Check permissions or disk space.")

def main():
    """Main function to execute fact-checking."""
    if not validate_inputs(GEMINI_API_KEY, SERPSTACK_API_KEY, arabic_claim):
        return

    # Call Gemini API
    gemini_response = call_gemini_api(GEMINI_API_KEY, system_prompt, user_prompt)
    if not gemini_response:
        logging.error("No valid Gemini response received. Exiting.")
        return

    # Parse Gemini response
    result, explanation, gemini_links = parse_response(gemini_response)

    # Search web for additional sources
    search_links = search_web_for_sources(arabic_claim, SERPSTACK_API_KEY)

    # Add context from recent news (confirming Tebboune is alive)
    context_links = [
        "https://www.yahoo.com/news/diplomatic-tensions-escalate-algeria-expels-130237586.html",
        "https://www.bbc.co.uk/news/articles/cpqjdq4z7lyo"
    ]
    search_links = list(dict.fromkeys(search_links + context_links))  # Deduplicate

    # Combine and deduplicate all links
    all_links = list(dict.fromkeys(gemini_links + search_links))

    # Print results
    print("\n✅ Arabic Fact-Check Result:\n")
    print(gemini_response)
    print(f"\n🟢 النتيجة: {result}")
    print(f"\n📖 الشرح: {explanation}")
    if all_links:
        print("\n🔗 روابط المصادر:")
        if gemini_links:
            print("  من Gemini:")
            for link in gemini_links:
                if link in all_links:
                    print(f"  - {link}")
        if search_links:
            print("  من البحث على الويب:")
            for link in search_links:
                if link in all_links:
                    print(f"  - {link}")
    else:
        print("\n🔗 روابط: لم يتم العثور على روابط.")

    # Save results
    file_path = os.path.join("fact_check_results", "fact_result_00.json")
    save_results(arabic_claim, result, explanation, gemini_links, search_links, "gemini-1.5-flash", file_path)

if __name__ == "__main__":
    main()