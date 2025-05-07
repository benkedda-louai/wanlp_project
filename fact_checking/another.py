import sys
import re
import json
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import requests
import os
import time

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: The 'google-generativeai' package is not installed.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# API keys configuration
API_KEYS = {
    "gemini": "AIzaSyBFIwjEW0AGIgF4U_9nvYSCiBs4lhU2pl8",
    "openrouter": "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"
}

# Model configurations
MODELS = {
    "gemini": {
        "name": "google/gemini-2.5-pro-preview-03-25",
        "provider": "google",
        "temperature": 0.3
    },
    "gpt-3.5": {
        "name": "openai/gpt-3.5-turbo",
        "provider": "openrouter",
        "temperature": 0.3
    },
    "claude-2": {
        "name": "anthropic/claude-3.5-haiku-20241022:beta",
        "provider": "openrouter",
        "temperature": 0.3
    },
    "claude-3": {
        "name": "anthropic/claude-3.7-sonnet",
        "provider": "openrouter",
        "temperature": 0.3
    },
    "llama-3": {
        "name": "meta-llama/llama-3-8b-instruct",
        "provider": "openrouter",
        "temperature": 0.3
    },
    "mistral": {
        "name": "mistralai/mistral-small-3.1-24b-instruct:free",
        "provider": "openrouter",
        "temperature": 0.3
    }
}

# System prompt template
SYSTEM_PROMPT = (
    "You are an expert Arabic fact-checker. Analyze the claim and provide:\n"
    "1. Verdict (صحيح/خاطئ/غير مؤكد)\n"
    "2. Detailed explanation in Arabic\n"
    "3. List of reliable source URLs that confirm your analysis\n"
    "Format your response clearly with these sections marked."
)

# Trusted news sources
TRUSTED_DOMAINS = [
    "misbar.com", "fatabyyano.net", "reuters.com", "bbc.com", "aljazeera.net",
    "aps.dz", "ennaharonline.com", "elwatan-dz.com", "alyaoum24.com",
    "hespress.com", "almayadeen.net", "arabic.cnn.com"
]

SOURCE_MAPPING = {
    "رويترز": "https://www.reuters.com",
    "وكالة الأنباء الجزائرية": "https://www.aps.dz",
    "النهار": "https://www.ennaharonline.com",
    "الوطن": "https://www.elwatan-dz.com",
    "مسبار": "https://www.misbar.com",
    "فتبينوا": "https://fatabyyano.net",
    "بي بي سي": "https://www.bbc.com/arabic",
    "الجزيرة": "https://www.aljazeera.net"
}

class FactChecker:
    def __init__(self):
        self.available_models = self._check_available_models()
        
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which models are available."""
        return {
            "gemini": True,
            "gpt-3.5": bool(API_KEYS.get("openrouter")),
            "claude-2": bool(API_KEYS.get("openrouter")),
            "claude-3": bool(API_KEYS.get("openrouter")),
            "llama-3": bool(API_KEYS.get("openrouter")),
            "mistral": bool(API_KEYS.get("openrouter"))
        }
    
    def translate_darija_to_msa(self, text: str) -> str:
        """Translate Algerian Darija to Modern Standard Arabic."""
        try:
            genai.configure(api_key=API_KEYS["gemini"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Translate this Algerian Darija to Modern Standard Arabic: {text}",
                generation_config={"temperature": 0.1}
            )
            if response.text:
                return self._clean_text(response.text)
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
        return text
    
    def fact_check(self, claim: str) -> Dict:
        """Perform fact-checking using all available models."""
        arabic_claim = self.translate_darija_to_msa(claim)
        results = {}
        
        for model_name, is_available in self.available_models.items():
            if not is_available:
                continue
                
            try:
                if model_name == "gemini":
                    response = self._call_gemini(arabic_claim)
                else:
                    response = self._call_openrouter(arabic_claim, model_name)
                
                if response:
                    verdict, explanation, sources = self.parse_response(response)
                    results[model_name] = {
                        "verdict": verdict,
                        "explanation": explanation,
                        "sources": sources,
                        "model_used": MODELS[model_name]["name"]
                    }
            except Exception as e:
                logging.error(f"Error with {model_name}: {str(e)}")
        
        return {
            "original_claim": claim,
            "translated_claim": arabic_claim,
            "results": results,
            "metadata": {
                "date": datetime.now().isoformat(),
                "models_used": [m["name"] for m in MODELS.values()]
            }
        }
    
    def _call_gemini(self, claim: str) -> Optional[str]:
        """Call Gemini API directly."""
        try:
            genai.configure(api_key=API_KEYS["gemini"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SYSTEM_PROMPT}\n\nClaim: {claim}"
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": MODELS["gemini"]["temperature"],
                    "max_output_tokens": 2000
                }
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return None
    
    def _call_openrouter(self, claim: str, model_key: str) -> Optional[str]:
        """Call model through OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {API_KEYS['openrouter']}",
                "HTTP-Referer": "https://github.com",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MODELS[model_key]["name"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Claim to verify: {claim}"}
                ],
                "temperature": MODELS[model_key]["temperature"],
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"OpenRouter error ({model_key}): {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"OpenRouter API error ({model_key}): {str(e)}")
            return None
    
    def parse_response(self, response: str) -> Tuple[str, str, List[str]]:
        """Parse model response into structured data."""
        # Extract verdict
        verdict_match = re.search(
            r"(?:النتيجة|Verdict|result)\s*[:]?\s*(صحيح|خاطئ|غير مؤكد|true|false|unverified)",
            response, re.IGNORECASE
        )
        verdict = verdict_match.group(1) if verdict_match else "غير مؤكد"
        
        # Extract explanation
        explanation_match = re.search(
            r"(?:الشرح|Explanation|analysis)\s*[:]?(.*?)(?=(?:المصادر|Sources|3\.|\n\n))",
            response, re.DOTALL
        )
        explanation = self._clean_text(explanation_match.group(1)) if explanation_match else "لا يوجد شرح"
        
        # Extract sources
        sources = []
        url_matches = re.findall(r'https?://[^\s<>"\']+', response)
        sources.extend([url for url in url_matches if any(domain in url for domain in TRUSTED_DOMAINS)])
        
        # Add sources mentioned in text
        for name, url in SOURCE_MAPPING.items():
            if name in response and url not in sources:
                sources.append(url)
        
        return verdict, explanation, list(dict.fromkeys(sources))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r'^\s*[\'"\s]+|[\'"\s]+\s*$', '', text.strip())
        return re.sub(r'\s+', ' ', text)
    
    def save_results(self, data: Dict, file_path: str) -> bool:
        """Save results to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Results saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            return False

def main():
    """Main execution function."""
    checker = FactChecker()
    
    # Get user input
    claim = input("أدخل الادعاء بالدارجة الجزائرية: ").strip()
    if not claim:
        print("لم يتم إدخال أي نص!")
        return
    
    print("\n🔍 جاري التحقق من صحة الادعاء باستخدام عدة نماذج...")
    results = checker.fact_check(claim)
    
    # Display results
    print("\n📋 النتائج:")
    print(f"الادعاء الأصلي: {results['original_claim']}")
    print(f"الترجمة للفصحى: {results['translated_claim']}")
    
    for model, data in results['results'].items():
        print(f"\n--- {model.upper()} ---")
        print(f"النتيجة: {data['verdict']}")
        print(f"التفسير: {data['explanation']}")
        if data['sources']:
            print("المصادر:")
            for src in data['sources']:
                print(f"- {src}")
    
    # Save to file
    output_path = r"./results/fact_check_results.json"
    if checker.save_results(results, output_path):
        print(f"\n💾 تم حفظ النتائج في: {output_path}")

if __name__ == "__main__":
    main()