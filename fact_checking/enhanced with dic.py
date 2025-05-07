import sys
import re
import json
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import requests
import os
import time
from urllib.parse import quote
import asyncio
import aiohttp

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: The 'google-generativeai' package is not installed.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)

# Import the Darija dictionary
from darija_dictionary import darija_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# API keys configuration
API_KEYS = {
    "gemini": "AIzaSyBFIwjEW0AGIgF4U_9nvYSCiBs4lhU2pl8",
    "openrouter": "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",
    "serpapi": "3f0582baad88dc99daa443f97e23aad1949aeb12"  
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
        "name": "anthropic/claude-3.7-sonnet:thinking",
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
    "You are an expert Arabic fact-checker specializing in Algerian news. Analyze the claim and provide:\n"
    "1. Verdict (صحيح/خاطئ/غير مؤكد)\n"
    "2. Detailed explanation in Arabic, including context and any regional nuances\n"
    "3. List of reliable source URLs that confirm your analysis, prioritizing Algerian and Arabic sources\n"
    "4. Confidence score (0-100%) based on evidence strength\n"
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
        self.rate_limit_delay = 2  # Delay between API calls in seconds
        
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
    
    async def translate_darija_to_msa(self, text: str) -> str:
        """Translate Algerian Darija to Modern Standard Arabic with context-aware preprocessing."""
        try:
            # Preprocess Darija to handle common expressions
            text = self._preprocess_darija(text)
            
            genai.configure(api_key=API_KEYS["gemini"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Translate this Algerian Darija to Modern Standard Arabic, preserving regional context: {text}",
                generation_config={"temperature": 0.1}
            )
            if response.text:
                return self._clean_text(response.text)
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
        return text
    
    def _preprocess_darija(self, text: str) -> str:
        """Preprocess Darija text to handle common expressions."""
        for darija, msa in darija_dict.items():
            text = re.sub(rf"\b{darija}\b", msa, text, flags=re.IGNORECASE)
        return text
    
    async def search_news(self, claim: str) -> List[str]:
        """Search for news articles related to the claim using SerpAPI."""
        try:
            async with aiohttp.ClientSession() as session:
                query = f"{quote(claim)} site:({' OR site:'.join(TRUSTED_DOMAINS)})"
                params = {
                    "q": query,
                    "api_key": API_KEYS.get("serpapi"),
                    "num": 10
                }
                async with session.get("https://serpapi.com/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [result["link"] for result in data.get("organic_results", [])]
        except Exception as e:
            logging.error(f"News search error: {str(e)}")
        return []
    
    async def fact_check(self, claim: str) -> Dict:
        """Perform fact-checking using all available models with enhanced search."""
        arabic_claim = await self.translate_darija_to_msa(claim)
        external_sources = await self.search_news(arabic_claim)
        results = {}
        
        for model_name, is_available in self.available_models.items():
            if not is_available:
                continue
                
            try:
                if model_name == "gemini":
                    response = await self._call_gemini(arabic_claim, external_sources)
                else:
                    response = await self._call_openrouter(arabic_claim, model_name, external_sources)
                
                if response:
                    verdict, explanation, sources, confidence = self.parse_response(response)
                    results[model_name] = {
                        "verdict": verdict,
                        "explanation": explanation,
                        "sources": sources,
                        "confidence": confidence,
                        "model_used": MODELS[model_name]["name"]
                    }
                await asyncio.sleep(self.rate_limit_delay)  # Respect API rate limits
            except Exception as e:
                logging.error(f"Error with {model_name}: {str(e)}")
        
        consensus = self._aggregate_results(results)
        return {
            "original_claim": claim,
            "translated_claim": arabic_claim,
            "results": results,
            "consensus": consensus,
            "metadata": {
                "date": datetime.now().isoformat(),
                "models_used": [m["name"] for m in MODELS.values()],
                "external_sources": external_sources
            }
        }
    
    async def _call_gemini(self, claim: str, external_sources: List[str]) -> Optional[str]:
        """Call Gemini API directly with external sources."""
        try:
            genai.configure(api_key=API_KEYS["gemini"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"{SYSTEM_PROMPT}\n\nClaim: {claim}\n"
                f"External Sources: {', '.join(external_sources) if external_sources else 'None'}"
            )
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": MODELS["gemini"]["temperature"],
                    "max_output_tokens": 3000
                }
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return None
    
    async def _call_openrouter(self, claim: str, model_key: str, external_sources: List[str]) -> Optional[str]:
        """Call model through OpenRouter API with external sources."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {API_KEYS['openrouter']}",
                    "HTTP-Referer": "https://github.com",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": MODELS[model_key]["name"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Claim to verify: {claim}\n"
                                f"External Sources: {', '.join(external_sources) if external_sources else 'None'}"
                            )
                        }
                    ],
                    "temperature": MODELS[model_key]["temperature"],
                    "max_tokens": 3000
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        logging.error(f"OpenRouter error ({model_key}): {await response.text()}")
                        return None
        except Exception as e:
            logging.error(f"OpenRouter API error ({model_key}): {str(e)}")
            return None
    
    def parse_response(self, response: str) -> Tuple[str, str, List[str], float]:
        """Parse model response into structured data with confidence score."""
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
        
        # Extract confidence score
        confidence_match = re.search(r"(?:الثقة|Confidence|score)\s*[:]?\s*(\d{1,3})", response)
        confidence = float(confidence_match.group(1)) if confidence_match else 50.0
        
        return verdict, explanation, list(dict.fromkeys(sources)), confidence
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results from multiple models to form a consensus."""
        if not results:
            return {"verdict": "غير مؤكد", "confidence": 0.0, "explanation": "لا توجد نتائج متاحة"}
        
        verdicts = [data["verdict"] for data in results.values()]
        confidences = [data["confidence"] for data in results.values()]
        explanations = [data["explanation"] for data in results.values()]
        
        # Determine majority verdict
        verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}
        majority_verdict = max(verdict_counts, key=verdict_counts.get)
        
        # Calculate weighted confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Combine explanations
        combined_explanation = "\n".join([f"{i+1}. {exp}" for i, exp in enumerate(explanations)])
        
        return {
            "verdict": majority_verdict,
            "confidence": avg_confidence,
            "explanation": combined_explanation
        }
    
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

async def main():
    """Main execution function."""
    checker = FactChecker()
    
    # Get user input
    claim = input("أدخل الادعاء بالدارجة الجزائرية: ").strip()
    if not claim:
        print("لم يتم إدخال أي نص!")
        return
    
    print("\n🔍 جاري التحقق من صحة الادعاء باستخدام عدة نماذج...")
    results = await checker.fact_check(claim)
    
    # Display results
    print("\n📋 النتائج:")
    print(f"الادعاء الأصلي: {results['original_claim']}")
    print(f"الترجمة للفصحى: {results['translated_claim']}")
    
    for model, data in results['results'].items():
        print(f"\n--- {model.upper()} ---")
        print(f"النتيجة: {data['verdict']}")
        print(f"الثقة: {data['confidence']}%")
        print(f"التفسير: {data['explanation']}")
        if data['sources']:
            print("المصادر:")
            for src in data['sources']:
                print(f"- {src}")
    
    # Display consensus
    print("\n--- الإجماع ---")
    print(f"النتيجة: {results['consensus']['verdict']}")
    print(f"الثقة: {results['consensus']['confidence']}%")
    print(f"التفسير: {results['consensus']['explanation']}")
    
    # Display external sources
    if results['metadata']['external_sources']:
        print("\nمصادر خارجية:")
        for src in results['metadata']['external_sources']:
            print(f"- {src}")
    
    # Save to file
    output_path = r"darija\fact_results_04.json"
    if checker.save_results(results, output_path):
        print(f"\n💾 تم حفظ النتائج في: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())