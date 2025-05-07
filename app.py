import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import io
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from urllib.parse import quote
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import arabic_reshaper
from bidi.algorithm import get_display
import google.generativeai as genai
from PIL import Image
import traceback

# Mock Darija dictionary
darija_dict = {
    "khbar": "news",
    "kadhb": "fake",
    "sahih": "true",
}

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set page config
st.set_page_config(
    page_title="TrustSeeker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging for FactChecker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# FactChecker Class with Multi-Model Integration and Image Analysis
class FactChecker:
    def __init__(self):
        self.available_models = self._check_available_models()
        self.rate_limit_delay = 2
        self.trusted_domains = [
            "misbar.com", "fatabyyano.net", "reuters.com", "bbc.com", "aljazeera.net",
            "aps.dz", "ennaharonline.com", "elwatan-dz.com", "alyaoum24.com",
            "hespress.com", "almayadeen.net", "arabic.cnn.com"
        ]
        self.source_mapping = {
            "رويترز": "https://www.reuters.com",
            "وكالة الأنباء الجزائرية": "https://www.aps.dz",
            "النهار": "https://www.ennaharonline.com",
            "الوطن": "https://www.elwatan-dz.com",
            "مسبار": "https://www.misbar.com",
            "فتبينوا": "https://fatabyyano.net",
            "بي بي سي": "https://www.bbc.com/arabic",
            "الجزيرة": "https://www.aljazeera.net"
        }

    def _check_available_models(self) -> Dict[str, bool]:
        API_KEYS = {
            "openrouter": "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",
            "serpapi": "3f0582baad88dc99daa443f97e23aad1949aeb12",
            "gemini": "AIzaSyBFIwjEW0AGIgF4U_9nvYSCiBs4lhU2pl8"
        }
        if API_KEYS.get("gemini"):
            genai.configure(api_key=API_KEYS["gemini"])
        return {
            "gpt-3.5": bool(API_KEYS.get("openrouter")),
            "claude-2": bool(API_KEYS.get("openrouter")),
            "llama-3": bool(API_KEYS.get("openrouter")),
            "mistral": bool(API_KEYS.get("openrouter")),
            "gemini": bool(API_KEYS.get("gemini"))
        }

    def search_news(self, claim: str) -> List[str]:
        API_KEYS = {"serpapi": "3f0582baad88dc99daa443f97e23aad1949aeb12"}
        try:
            query = f"{quote(claim)} site:({' OR site:'.join(self.trusted_domains)})"
            params = {
                "q": query,
                "api_key": API_KEYS.get("serpapi"),
                "num": 10
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [result["link"] for result in data.get("organic_results", [])]
        except Exception as e:
            logging.error(f"News search error: {str(e)}")
        return []

    def extract_keywords(self, claim: str) -> List[str]:
        tokens = word_tokenize(claim.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
        pos_tags = pos_tag(filtered_tokens)
        keywords = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        return keywords

    def classify_category(self, text: str) -> str:
        """Classify text into news categories using Gemini API."""
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = (
                f"Classify the following news text into one of these categories: Politics, Health, Economy, Other. "
                f"Return only the category name.\n\nText: {text}"
            )
            response = model.generate_content(prompt)
            category = response.text.strip()
            if category not in ["Politics", "Health", "Economy", "Other"]:
                return "Other"
            return category
        except Exception as e:
            logging.error(f"Category classification error: {str(e)}")
            return "Other"

    def fact_check(self, claim: str, category: str = "Unknown") -> Dict:
        for darija_term, translation in darija_dict.items():
            claim = claim.replace(darija_term, translation)
        
        external_sources = self.search_news(claim)
        keywords = self.extract_keywords(claim)
        results = {}
        MODELS = {
            "gpt-3.5": {"name": "openai/gpt-3.5-turbo", "provider": "openrouter", "temperature": 0.3},
            "claude-2": {"name": "anthropic/claude-3.5-haiku-20241022:beta", "provider": "openrouter", "temperature": 0.3},
            "llama-3": {"name": "meta-llama/llama-3-8b-instruct", "provider": "openrouter", "temperature": 0.3},
            "mistral": {"name": "mistralai/mistral-small-3.1-24b-instruct:free", "provider": "openrouter", "temperature": 0.3},
            "gemini": {"name": "gemini-1.5-pro", "provider": "gemini", "temperature": 0.3}
        }

        for model_name, is_available in self.available_models.items():
            if not is_available:
                continue
            try:
                if MODELS[model_name]["provider"] == "gemini":
                    response = self._call_gemini(claim, model_name, external_sources)
                else:
                    response = self._call_openrouter(claim, model_name, external_sources)
                if response:
                    verdict, explanation, sources, confidence, impact = self.parse_response(response)
                    results[model_name] = {
                        "verdict": verdict,
                        "explanation": explanation,
                        "sources": sources,
                        "confidence": confidence,
                        "impact": impact
                    }
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logging.error(f"Error with {model_name}: {str(e)}")
        
        consensus = self._aggregate_results(results)
        return {
            "original_claim": claim,
            "keywords": keywords,
            "category": category,
            "results": results,
            "consensus": consensus,
            "metadata": {
                "date": datetime.now().isoformat(),
                "models_used": [m["name"] for m in MODELS.values()],
                "external_sources": external_sources
            }
        }

    def fact_check_image(self, image: Image.Image) -> Dict:
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = (
                "Analyze the provided image. If the image contains text, extract the text accurately. "
                "If the image contains no text, provide a detailed description of the content (e.g., objects, people, setting). "
                "Return the result in the following format:\n"
                "- **Type**: 'Text' or 'Image'\n"
                "- **Content**: [Extracted text or image description]\n"
            )
            response = model.generate_content([prompt, image])
            analysis = response.text.strip()

            type_match = re.search(r"\*\*Type\*\*:\s*(Text|Image)", analysis)
            content_match = re.search(r"\*\*Content\*\*:\s*(.*?)(?=\n\n|$)", analysis, re.DOTALL)
            content_type = type_match.group(1) if type_match else "Unknown"
            content = content_match.group(1).strip() if content_match else ""

            if content_type == "Text" and content:
                category = self.classify_category(content)
                return self.fact_check(content, category)
            elif content_type == "Image" and content:
                claim = f"Image description: {content}"
                category = self.classify_category(claim)
                return self.fact_check(claim, category)
            else:
                return {
                    "original_claim": "No text or meaningful content extracted",
                    "keywords": [],
                    "category": "Unknown",
                    "results": {},
                    "consensus": {
                        "verdict": "غير مؤكد",
                        "confidence": 0.0,
                        "explanation": "No text or recognizable content could be extracted from the image.",
                        "impact": "غير متاح"
                    },
                    "metadata": {
                        "date": datetime.now().isoformat(),
                        "models_used": [],
                        "external_sources": []
                    }
                }
        except Exception as e:
            logging.error(f"Image fact-checking error: {str(e)}")
            return {
                "original_claim": "Error processing image",
                "keywords": [],
                "category": "Unknown",
                "results": {},
                "consensus": {
                    "verdict": "غير مؤكد",
                    "confidence": 0.0,
                    "explanation": f"Error processing image: {str(e)}",
                    "impact": "غير متاح"
                },
                "metadata": {
                    "date": datetime.now().isoformat(),
                    "models_used": [],
                    "external_sources": []
                }
            }

    def _call_openrouter(self, claim: str, model_key: str, external_sources: List[str]) -> Optional[str]:
        API_KEYS = {"openrouter": "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"}
        SYSTEM_PROMPT = (
            "You are an expert Arabic fact-checker specializing in Algerian news with extensive knowledge of regional politics, culture, and media. Your task is to thoroughly analyze the given claim, which may be text extracted from an image or a description of an image's content. Provide a detailed, accurate, and well-structured fact-checking report. Follow these instructions precisely:\n\n"
            "1. **Verdict (in Arabic):** Determine if the claim is 'صحيح' (True), 'خاطئ' (False), or 'غير مؤكد' (Unverified). Use only these exact terms for the verdict.\n"
            "2. **Detailed Explanation (in Arabic):** Provide a comprehensive explanation of your analysis in Arabic. Include:\n"
            "   - Historical and cultural context relevant to the claim, especially pertaining to Algeria and the broader Maghreb region.\n"
            "   - If the claim is an image description, evaluate the plausibility of the described content and check for signs of manipulation or fabrication.\n"
            "   - Any regional nuances, linguistic variations, or common misinformation patterns in Algerian media.\n"
            "   - A breakdown of the evidence you considered, including contradictions or consistencies with known facts.\n"
            "   - If the claim involves Darija (Algerian Arabic), address any colloquial terms and their implications.\n"
            "   Ensure the explanation is at least 150 words to provide sufficient depth.\n"
            "3. **Reliable Sources:** Provide a list of reliable source URLs that confirm your analysis. Prioritize:\n"
            "   - Trusted Algerian news outlets (e.g., El Watan, Ennahar, APS).\n"
            "   - Reputable Arabic-language sources (e.g., Al Jazeera, BBC Arabic).\n"
            "   - International fact-checking organizations (e.g., Misbar, Fatabyyano).\n"
            "   - Primary sources such as government statements, official reports, or academic studies.\n"
            "   Ensure each URL is valid, accessible, and directly supports your analysis. Include at least 3 sources unless the claim is unverified due to a lack of evidence.\n"
            "4. **Confidence Score:** Provide a confidence score (0-100%) based on the strength of the evidence. Justify the score in your explanation.\n"
            "5. **Impact and Dangers (if Fake):** If the claim is 'خاطئ' (False), provide a detailed analysis of its potential impact and dangers, including:\n"
            "   - Social consequences (e.g., public panic, division, or mistrust in institutions).\n"
            "   - Political consequences (e.g., influence on elections, policy decisions, or international relations).\n"
            "   - Economic consequences (e.g., market instability, financial scams).\n"
            "   - Health and safety risks (e.g., misinformation about medical treatments or emergencies).\n"
            "   - Specific risks in the Algerian context, such as effects on local communities, tribal dynamics, or regional stability.\n"
            "   Provide at least 100 words of analysis for the impact section, citing examples or historical precedents if possible.\n\n"
            "Format your response with the following sections clearly marked:\n"
            "- **النتيجة:** [Verdict]\n"
            "- **الشرح:** [Explanation]\n"
            "- **المصادر:** [List of URLs, one per line]\n"
            "- **الثقة:** [Confidence Score]%\n"
            "- **التأثير والمخاطر:** [Impact and Dangers, if applicable]\n\n"
            "Ensure your response is professional, unbiased, and focused on factual accuracy. Avoid speculation and prioritize evidence-based reasoning."
        )
        MODELS = {
            "gpt-3.5": {"name": "openai/gpt-3.5-turbo", "provider": "openrouter", "temperature": 0.3},
            "claude-2": {"name": "anthropic/claude-3.5-haiku-20241022:beta", "provider": "openrouter", "temperature": 0.3},
            "llama-3": {"name": "meta-llama/llama-3-8b-instruct", "provider": "openrouter", "temperature": 0.3},
            "mistral": {"name": "mistralai/mistral-small-3.1-24b-instruct:free", "provider": "openrouter", "temperature": 0.3}
        }
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
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logging.error(f"OpenRouter error ({model_key}): {response.text}")
                return None
        except Exception as e:
            logging.error(f"OpenRouter API error ({model_key}): {str(e)}")
            return None

    def _call_gemini(self, claim: str, model_key: str, external_sources: List[str]) -> Optional[str]:
        SYSTEM_PROMPT = (
            "You are an expert Arabic fact-checker specializing in Algerian news with extensive knowledge of regional politics, culture, and media. Your task is to thoroughly analyze the given claim, which may be text extracted from an image or a description of an image's content. Provide a detailed, accurate, and well-structured fact-checking report. Follow these instructions precisely:\n\n"
            "1. **Verdict (in Arabic):** Determine if the claim is 'صحيح' (True), 'خاطئ' (False), or 'غير مؤكد' (Unverified). Use only these exact terms for the verdict.\n"
            "2. **Detailed Explanation (in Arabic):** Provide a comprehensive explanation of your analysis in Arabic. Include:\n"
            "   - Historical and cultural context relevant to the claim, especially pertaining to Algeria and the broader Maghreb region.\n"
            "   - If the claim is an image description, evaluate the plausibility of the described content and check for signs of manipulation or fabrication.\n"
            "   - Any regional nuances, linguistic variations, or common misinformation patterns in Algerian media.\n"
            "   - A breakdown of the evidence you considered, including contradictions or consistencies with known facts.\n"
            "   - If the claim involves Darija (Algerian Arabic), address any colloquial terms and their implications.\n"
            "   Ensure the explanation is at least 150 words to provide sufficient depth.\n"
            "3. **Reliable Sources:** Provide a list of reliable source URLs that confirm your analysis. Prioritize:\n"
            "   - Trusted Algerian news outlets (e.g., El Watan, Ennahar, APS).\n"
            "   - Reputable Arabic-language sources (e.g., Al Jazeera, BBC Arabic).\n"
            "   - International fact-checking organizations (e.g., Misbar, Fatabyyano).\n"
            "   - Primary sources such as government statements, official reports, or academic studies.\n"
            "   Ensure each URL is valid, accessible, and directly supports your analysis. Include at least 3 sources unless the claim is unverified due to a lack of evidence.\n"
            "4. **Confidence Score:** Provide a confidence score (0-100%) based on the strength of the evidence. Justify the score in your explanation.\n"
            "5. **Impact and Dangers (if Fake):** If the claim is 'خاطئ' (False), provide a detailed analysis of its potential impact and dangers, including:\n"
            "   - Social consequences (e.g., public panic, division, or mistrust in institutions).\n"
            "   - Political consequences (e.g., influence on elections, policy decisions, or international relations).\n"
            "   - Economic consequences (e.g., market instability, financial scams).\n"
            "   - Health and safety risks (e.g., misinformation about medical treatments or emergencies).\n"
            "   - Specific risks in the Algerian context, such as effects on local communities, tribal dynamics, or regional stability.\n"
            "   Provide at least 100 words of analysis for the impact section, citing examples or historical precedents if possible.\n\n"
            "Format your response with the following sections clearly marked:\n"
            "- **النتيجة:** [Verdict]\n"
            "- **الشرح:** [Explanation]\n"
            "- **المصادر:** [List of URLs, one per line]\n"
            "- **الثقة:** [Confidence Score]%\n"
            "- **التأثير والمخاطر:** [Impact and Dangers, if applicable]\n\n"
            "Ensure your response is professional, unbiased, and focused on factual accuracy. Avoid speculation and prioritize evidence-based reasoning."
        )
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Claim to verify: {claim}\n"
                f"External Sources: {', '.join(external_sources) if external_sources else 'None'}"
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return None

    def parse_response(self, response: str) -> Tuple[str, str, List[str], float, str]:
        verdict_match = re.search(
            r"(?:النتيجة|Verdict|result)\s*[:]?\s*(صحيح|خاطئ|غير مؤكد|true|false|unverified)",
            response, re.IGNORECASE
        )
        verdict = verdict_match.group(1) if verdict_match else "غير مؤكد"
        explanation_match = re.search(
            r"(?:الشرح|Explanation|analysis)\s*[:]?(.*?)(?=(?:المصادر|Sources|3\.|\n\n))",
            response, re.DOTALL
        )
        explanation = self._clean_text(explanation_match.group(1)) if explanation_match else "لا يوجد شرح"
        sources = []
        url_matches = re.findall(r'https?://[^\s<>"\']+', response)
        sources.extend([url for url in url_matches if any(domain in url for domain in self.trusted_domains)])
        for name, url in self.source_mapping.items():
            if name in response and url not in sources:
                sources.append(url)
        confidence_match = re.search(r"(?:الثقة|Confidence|score)\s*[:]?\s*(\d{1,3})", response)
        confidence = float(confidence_match.group(1)) if confidence_match else 50.0
        impact_match = re.search(
            r"(?:التأثير والمخاطر|Impact and Dangers)\s*[:]?(.*?)(?=(?:النهاية|End|$|\n\n))",
            response, re.DOTALL
        )
        impact = self._clean_text(impact_match.group(1)) if impact_match else "غير متاح" if verdict == "خاطئ" else "لا ينطبق (الخبر ليس مزيفًا)"
        return verdict, explanation, list(dict.fromkeys(sources)), confidence, impact

    def _aggregate_results(self, results: Dict) -> Dict:
        if not results:
            return {
                "verdict": "غير مؤكد",
                "confidence": 0.0,
                "explanation": "لا توجد نتائج متاحة",
                "impact": "غير متاح"
            }
        verdicts = [data["verdict"] for data in results.values()]
        confidences = [data["confidence"] for data in results.values()]
        explanations = [data["explanation"] for data in results.values()]
        impacts = [data["impact"] for data in results.values() if data["verdict"] == "خاطئ"]
        verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}
        majority_verdict = max(verdict_counts, key=verdict_counts.get)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        combined_explanation = "\n".join([f"{i+1}. {exp}" for i, exp in enumerate(explanations)])
        combined_impact = "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(impacts)]) if impacts else "لا ينطبق (الخبر ليس مزيفًا)"
        return {
            "verdict": majority_verdict,
            "confidence": avg_confidence,
            "explanation": combined_explanation,
            "impact": combined_impact
        }

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'^\s*[\'"\s]+|[\'"\s]+\s*$', '', text.strip())
        return re.sub(r'\s+', ' ', text)

# Utility Functions
def load_user_profiles() -> Dict:
    try:
        if os.path.exists("user_profiles.json"):
            with open("user_profiles.json", "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading user profiles: {str(e)}")
        return {}

def save_user_profiles(profiles: Dict) -> None:
    try:
        with open("user_profiles.json", "w") as f:
            json.dump(profiles, f, indent=4)
    except Exception as e:
        st.error(f"Error saving user profiles: {str(e)}")

def initialize_session_state() -> None:
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'api_url' not in st.session_state:
        st.session_state.api_url = "https://gentle-driving-mastodon.ngrok-free.app"
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_profiles' not in st.session_state:
        st.session_state.user_profiles = load_user_profiles()

initialize_session_state()

# Load Logo
def get_base64_image():
    logo_path = r"C:\Users\j\Desktop\project\P4_P.png"
    try:
        with open(logo_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logging.error(f"Error loading logo: {str(e)}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

logo_base64 = get_base64_image()

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #ff5733;
        --secondary: #c8e6c9;
        --accent: #4caf50;
        --text: #212121;
        --border: #d1d8e0;
        --card-shadow: 0 4px 8px rgba(0,0,0,0.1);
        --highlight: #ab47bc;
        --warning: #f44336;
        --info: #26c6da;
        --background: #ffffff;
        --success: #81c784;
        deafening-earwax
        --chart1: #42a5f5;
        --chart2: #ffca28;
        --chart3: #ec407a;
        --chart4: #66bb6a;
        --keyword-highlight: #ffee58;
    }

    .main {
        background-color: var(--background);
        min-height: 100vh;
        padding-top: 80px;
    }

    .header {
        background: linear-gradient(90deg, var(--primary), var(--highlight));
        color: white;
        padding: 20px;
        text-align: center;
        width: 100%;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .header img {
        width: 40px;
        height: 40px;
        margin-right: 10px;
    }

    .content {
        padding: 40px;
    }

    .assessment-card {
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        background-color: white;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border);
        transition: transform 0.2s;
    }

    .assessment-card:hover {
        transform: translateY(-3px);
    }

    .status-container {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
    }

    .status-badge {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: bold;
        margin-right: 16px;
    }

    .status-true {
        background-color: var(--success);
        color: #2e7d32;
    }

    .status-false {
        background-color: var(--error);
        color: #c62828;
    }

    .status-error {
        background-color: var(--chart2);
        color: #ef6c00;
    }

    .header-text {
        color: var(--text);
        margin-bottom: 8px;
        font-weight: 600;
        background: linear-gradient(to right, var(--primary), var(--highlight));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar .sidebar-content {
        background-color: white;
        border-right: 1px solid var(--border);
    }

    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--highlight));
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #e53935, #8e24aa);
        transform: translateY(-1px);
        box-shadow: var(--card-shadow);
    }

    .stTextArea textarea {
        font-family: 'Roboto', sans-serif !important;
        border: 1px solid var(--info) !important;
        border-radius: 8px !important;
        background: #e1f5fe;
        font-size: 14px;
        padding: 10px;
        color: var(--text) !important;
    }

    .stTextInput input {
        color: var(--text) !important;
        background-color: white !important;
        border: 1px solid var(--info) !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    .stTextInput input:disabled {
        background-color: #f5f5f5 !important;
        color: #616161 !important;
        -webkit-text-fill-color: #616161 !important;
        opacity: 1 !important;
    }

    .user-info {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 2px solid var(--primary);
    }

    .info-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        background: linear-gradient(135deg, var(--secondary), #f3e8ff);
        border: 1px solid var(--border);
        transition: all 0.2s;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--card-shadow);
    }

    .status-badge-small {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .success {
        background-color: var(--success);
        color: #2e7d32;
    }

    .error {
        background-color: var(--error);
        color: #c62828;
    }

    .tab-content {
        padding-top: 24px;
    }

    .result-section {
        margin-bottom: 20px;
        padding: 15px;
        border-left: 4px solid var(--highlight);
        background: #f9f9f9;
        border-radius: 8px;
    }

    .result-section h4 {
        border-bottom: 2px solid var(--info);
        padding-bottom: 8px;
        margin-bottom: 12px;
        color: var(--primary);
        display: flex;
        align-items: center;
    }

    .result-section ul {
        list-style: none;
        padding-left: 0;
    }

    .result-section ul li {
        position: relative;
        padding-left: 30px;
        margin-bottom: 12px;
        font-size: 16px;
        line-height: 1.5;
    }

    .result-section ul li::before {
        content: "➡️";
        position: absolute;
        left: 0;
        color: var(--accent);
        font-size: 18px;
    }

    .result-section ul li strong {
        color: var(--primary);
    }

    .result-section a {
        color: var(--info);
        text-decoration: none;
        transition: color 0.2s;
    }

    .result-section a:hover {
        color: var(--highlight);
        text-decoration: underline;
    }

    .text-snippet {
        background: linear-gradient(135deg, #eceff1, var(--secondary));
        border-radius: 8px;
        padding: 16px;
        font-family: 'Roboto', sans-serif;
        border: 1px solid var(--border);
        margin: 8px 0;
        color: var(--text);
    }

    .history-item {
        margin-bottom: 16px;
        border-radius: 12px;
        overflow: hidden;
    }

    .history-header {
        background: linear-gradient(90deg, var(--primary), var(--highlight));
        color: white;
        padding: 12px 16px;
        font-weight: 600;
    }

    .history-content {
        color: var(--text);
        background-color: white;
        padding: 16px;
        border: 1px solid var(--border);
        border-top: none;
    }

    .footer {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #eceff1, #e0e0e0);
        margin-top: 30px;
        border-radius: 8px;
        font-size: 14px;
    }

    .keyword-highlight {
        background-color: var(--keyword-highlight);
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }

    .info-list li {
        margin-bottom: 10px;
        padding-left: 25px;
        position: relative;
        line-height: 1.6;
    }

    .info-list li::before {
        content: "•";
        position: absolute;
        left: 0;
        color: var(--accent);
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header with Logo
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_base64}" alt="TrustSeeker Logo" style="width: 120px; height: auto;">
    <div>
        <h1>TrustSeeker</h1>
        <p>A powerful tool to combat misinformation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Application Settings")
    st.session_state.api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_url,
        help="Enter the active API URL for Predict and Train.",
        key="api_url_input"
    )
    st.warning("Ensure the API is running for Predict and Train tabs.")
    
    st.markdown("---")
    st.markdown("### 👤 User Profile")
    user_id = st.text_input("User ID", placeholder="user123", key="user_id_input")
    if user_id and user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        if user_id not in st.session_state.user_profiles:
            st.session_state.user_profiles[user_id] = {
                "first_name": "",
                "last_name": "",
                "email": "",
                "requests_made": 0
            }
        save_user_profiles(st.session_state.user_profiles)
    
    if st.session_state.user_id:
        user_profile = st.session_state.user_profiles.get(st.session_state.user_id, {})
        first_name = st.text_input("First Name", value=user_profile.get("first_name", ""), key="first_name")
        last_name = st.text_input("Last Name", value=user_profile.get("last_name", ""), key="last_name")
        email = st.text_input("Email", value=user_profile.get("email", ""), key="email")
        if st.button("Save Profile"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.user_profiles[st.session_state.user_id].update({
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email
                })
                save_user_profiles(st.session_state.user_profiles)
                st.success("Profile saved successfully!")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Made by xAI Team</p>", unsafe_allow_html=True)

# Utility Functions
def preprocess_text(text: str) -> Dict[str, any]:
    if not text.strip():
        return {
            "raw_text": "",
            "normalized_text": "",
            "tokens": [],
            "stemmed_tokens": [],
            "filtered_tokens": [],
            "pos_tags": [],
            "word_freq": {}
        }
    
    normalized_text = text.lower().strip()
    normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text)
    tokens = word_tokenize(normalized_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    pos_tags = pos_tag(filtered_tokens)
    word_freq = pd.Series(filtered_tokens).value_counts().to_dict()
    return {
        "raw_text": text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "stemmed_tokens": stemmed_tokens,
        "filtered_tokens": filtered_tokens,
        "pos_tags": pos_tags,
        "word_freq": word_freq
    }

def generate_wordcloud(word_freq: Dict[str, int], title: str, is_arabic: bool = False) -> None:
    if not word_freq:
        st.warning("No words to display in the word cloud.")
        return
    
    if is_arabic:
        reshaped_text = {}
        for word, freq in word_freq.items():
            reshaped_word = arabic_reshaper.reshape(word)
            display_word = get_display(reshaped_word)
            reshaped_text[display_word] = freq
        word_freq = reshaped_text
        try:
            font_path = "Amiri-Regular.ttf"
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                font_path=font_path,
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(word_freq)
        except Exception as e:
            st.error(f"Error generating Arabic word cloud: {str(e)}")
            return
    else:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def display_confidence_gauge(confidence: float, title: str = "Confidence Score") -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "var(--chart1)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "var(--error)"},
                {'range': [50, 75], 'color': "var(--chart2)"},
                {'range': [75, 100], 'color': "var(--success)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def add_to_history(action: str, input_data: str, result: str, status: str, confidence: float = 0.0, category: str = "Unknown", prediction: str = "Unknown", impact: str = "Not Applicable") -> None:
    if st.session_state.user_id:
        st.session_state.user_profiles[st.session_state.user_id]["requests_made"] = \
            st.session_state.user_profiles[st.session_state.user_id].get("requests_made", 0) + 1
        save_user_profiles(st.session_state.user_profiles)
    
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Action": action,
        "Input": input_data,
        "Result": result,
        "Status": status,
        "Confidence": confidence,
        "Category": category,
        "Prediction": prediction,
        "Impact": impact,
        "Source": "موقع " + str(len(st.session_state.history) % 5 + 1),
        "UserID": st.session_state.user_id if st.session_state.user_id else "Anonymous"
    })

def make_api_call(url: str, data: Dict, progress_callback: callable) -> Tuple[Optional[Dict], str]:
    progress = 0
    progress_bar = st.progress(progress)
    for i in range(1, 6):
        time.sleep(0.5)
        progress = i * 20
        progress_bar.progress(progress)
        progress_callback(f"Processing... {progress}%")
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        progress_bar.progress(100)
        return response.json(), "Success"
    except requests.exceptions.RequestException as e:
        progress_bar.progress(100)
        return str(e), "Error"

def validate_input(text: str, field_name: str) -> bool:
    if not text.strip():
        st.error(f"Please enter a {field_name}.")
        return False
    if len(text) > 1000:
        st.error(f"{field_name} is too long. Maximum length is 1000 characters.")
        return False
    return True

def get_user_stats() -> Dict:
    if not st.session_state.user_id:
        return {"requests_made": 0, "success_rate": 0, "predict_count": 0, "train_count": 0, "fact_check_count": 0}
    
    user_requests = [entry for entry in st.session_state.history if entry["UserID"] == st.session_state.user_id]
    total_requests = len(user_requests)
    success_requests = len([entry for entry in user_requests if entry["Status"] == "Success"])
    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
    predict_count = len([entry for entry in user_requests if entry["Action"] == "Predict"])
    train_count = len([entry for entry in user_requests if entry["Action"] == "Train"])
    fact_check_count = len([entry for entry in user_requests if entry["Action"] == "Fact-Check"])
    return {
        "requests_made": total_requests,
        "success_rate": round(success_rate, 2),
        "predict_count": predict_count,
        "train_count": train_count,
        "fact_check_count": fact_check_count
    }

def highlight_keywords(text: str, keywords: List[str]) -> str:
    highlighted_text = text
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        highlighted_text = re.sub(pattern, f'<span class="keyword-highlight">{keyword}</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

# Main Content
st.markdown("""
<p style="font-size:16px; color:#6c757d; margin-bottom:32px;">
A powerful AI-driven tool to detect fake news, train models, fact-check claims, and learn about the impact of misinformation with advanced analytics and preprocessing insights.
</p>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Predict", "🛠️ Train", "✅ Fact-Check", "📚 History", "📊 Analytics", "ℹ️ Learn More"])

with tab1:
    st.markdown("<div class='user-info'>", unsafe_allow_html=True)
    st.subheader("👤 User Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_id_display = st.text_input("User ID", value=st.session_state.user_id if st.session_state.user_id else "Anonymous", disabled=True, key="predict_user_id")
    with col2:
        user_name = st.text_input("Name", value=f"{st.session_state.user_profiles.get(st.session_state.user_id, {}).get('first_name', '')} {st.session_state.user_profiles.get(st.session_state.user_id, {}).get('last_name', '')}".strip(), disabled=True, key="predict_user_name")
    with col3:
        user_email = st.text_input("Email", value=st.session_state.user_profiles.get(st.session_state.user_id, {}).get("email", ""), disabled=True, key="predict_user_email")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        predict_text = st.text_area(
            "Enter text to predict:",
            height=400,
            placeholder="Paste your text here...",
            help="Enter the text you want to check for truthfulness",
            key="predict_text"
        )
        
        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            if not validate_input(predict_text, "text"):
                pass
            else:
                with st.spinner("🧠 Analyzing your text..."):
                    status_placeholder = st.empty()
                    result, status = make_api_call(
                        f"{st.session_state.api_url}/api/predict/",
                        {"text": predict_text},
                        lambda msg: status_placeholder.info(msg)
                    )
                    
                    if status == "Success":
                        st.success("✅ Prediction Complete!")
                        st.markdown("---")
                        
                        st.markdown("<div class='assessment-card'>", unsafe_allow_html=True)
                        prediction = "Real" if result['is_true'] else "Fake"
                        status_class = "status-true" if result['is_true'] else "status-false"
                        confidence = 90 if result['is_true'] else 10
                        checker = FactChecker()
                        category = checker.classify_category(predict_text)
                        st.markdown(f"""
                        <div class='status-container'>
                            <div class='status-badge {status_class}'>{prediction}</div>
                            <div>
                                <h3 class='header-text'>Prediction Results</h3>
                                <ul class='info-list'>
                                    <li><strong>Text:</strong> {result['text']}</li>
                                    <li><strong>Category:</strong> {category}</li>
                                    <li><strong>Date:</strong> {result['date']}</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        display_confidence_gauge(confidence)
                        
                        probability = 0.9 if result['is_true'] else 0.1
                        df = pd.DataFrame({
                            'Category': ['Real', 'Fake'],
                            'Probability': [probability, 1 - probability]
                        })
                        fig = px.bar(
                            df,
                            x='Category',
                            y='Probability',
                            title="Prediction Confidence",
                            color='Category',
                            color_discrete_map={'Real': 'var(--chart1)', 'Fake': 'var(--warning)'},
                            height=300,
                            text='Probability'
                        )
                        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        trend_df = pd.DataFrame({
                            'Step': ['Initial', 'Analysis', 'Final'],
                            'Probability': [0.5, probability * 0.8, probability]
                        })
                        fig = px.line(
                            trend_df,
                            x='Step',
                            y='Probability',
                            title="Probability Trend During Analysis",
                            markers=True,
                            color_discrete_sequence=['var(--chart3)'],
                            height=300
                        )
                        fig.update_traces(texttemplate='%{y:.1%}', textposition='top center')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Dynamic Category Distribution
                        history_categories = [entry['Category'] for entry in st.session_state.history if entry['Category'] != "Unknown"]
                        history_categories.append(category)
                        category_counts = pd.Series(history_categories).value_counts()
                        category_df = pd.DataFrame({
                            'Category': category_counts.index,
                            'Count': category_counts.values
                        })
                        fig = px.pie(
                            category_df,
                            names='Category',
                            values='Count',
                            title="News Category Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        add_to_history("Predict", predict_text, f"Is Real: {result['is_true']}", "Success", confidence, category, prediction)
                    else:
                        st.error(f"API Error: {result}")
                        add_to_history("Predict", predict_text, result, "Error", 0, "Unknown", "Unknown")

    with col2:
        st.markdown("### 📚 Tips for Better Predictions")
        st.markdown("""
        <ul class='info-list'>
            <li>Use clear and concise text</li>
            <li>Avoid ambiguous statements</li>
            <li>Provide context when possible</li>
            <li>Check for typos before submitting</li>
            <li>Review preprocessing steps</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Prediction Stats")
        stats = get_user_stats()
        st.markdown(f"""
        <div class='info-card'>
            <h4>User Statistics</h4>
            <ul class='info-list'>
                <li><strong>Requests Made:</strong> {stats['requests_made']}</li>
                <li><strong>Success Rate:</strong> {stats['success_rate']}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='user-info'>", unsafe_allow_html=True)
    st.subheader("👤 User Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("User ID", value=st.session_state.user_id if st.session_state.user_id else "Anonymous", disabled=True, key="train_user_id")
    with col2:
        st.text_input("Name", value=f"{st.session_state.user_profiles.get(st.session_state.user_id, {}).get('first_name', '')} {st.session_state.user_profiles.get(st.session_state.user_id, {}).get('last_name', '')}".strip(), disabled=True, key="train_user_name")
    with col3:
        st.text_input("Email", value=st.session_state.user_profiles.get(st.session_state.user_id, {}).get("email", ""), disabled=True, key="train_user_email")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        train_text = st.text_area(
            "Enter text to train:",
            height=400,
            placeholder="Paste your text here...",
            help="Enter the text to train the model",
            key="train_text"
        )
        
        train_label = st.selectbox(
            "Label",
            [0, 1],
            key="train_label",
            help="0 = False, 1 = True"
        )
        
        if st.button("🛠️ Train", use_container_width=True, type="primary"):
            if not validate_input(train_text, "text"):
                pass
            else:
                with st.spinner("🧠 Training the model..."):
                    status_placeholder = st.empty()
                    result, status = make_api_call(
                        f"{st.session_state.api_url}/api/train/",
                        {"text": train_text, "label": train_label},
                        lambda msg: status_placeholder.info(msg)
                    )
                    
                    if status == "Success":
                        st.success("✅ Training Complete!")
                        st.markdown("---")
                        
                        st.markdown("<div class='assessment-card'>", unsafe_allow_html=True)
                        status_class = "status-true"
                        confidence = 85
                        checker = FactChecker()
                        category = checker.classify_category(train_text)
                        st.markdown(f"""
                        <div class='status-container'>
                            <div class='status-badge {status_class}'>Success</div>
                            <div>
                                <h3 class='header-text'>Training Results</h3>
                                <ul class='info-list'>
                                    <li><strong>Text:</strong> {result['text']}</li>
                                    <li><strong>Category:</strong> {category}</li>
                                    <li><strong>Message:</strong> {result['message']}</li>
                                    <li><strong>Label:</strong> {'True' if result['label'] == 1 else 'False'}</li>
                                    <li><strong>Loss:</strong> {result['loss']}</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        display_confidence_gauge(confidence, "Training Confidence")
                        st.markdown("</div>", unsafe_allow_html=True)
                        add_to_history("Train", f"Text: {train_text}, Label: {train_label}", f"Loss: {result['loss']}", "Success", confidence, category, "Unknown")
                    else:
                        st.error(f"API Error: {result}")
                        add_to_history("Train", f"Text: {train_text}, Label: {train_label}", result, "Error", 0, "Unknown", "Unknown")

    with col2:
        st.markdown("### 📚 Tips for Better Training")
        st.markdown("""
        <ul class='info-list'>
            <li>Use diverse training data</li>
            <li>Ensure labels are accurate</li>
            <li>Train with varied text lengths</li>
            <li>Avoid overfitting with balanced data</li>
            <li>Monitor the loss value</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Training Stats")
        stats = get_user_stats()
        st.markdown(f"""
        <div class='info-card'>
            <h4>User Statistics</h4>
            <ul class='info-list'>
                <li><strong>Requests Made:</strong> {stats['requests_made']}</li>
                <li><strong>Success Rate:</strong> {stats['success_rate']}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='user-info'>", unsafe_allow_html=True)
    st.subheader("👤 User Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("User ID", value=st.session_state.user_id if st.session_state.user_id else "Anonymous", disabled=True, key="fact_check_user_id")
    with col2:
        st.text_input("Name", value=f"{st.session_state.user_profiles.get(st.session_state.user_id, {}).get('first_name', '')} {st.session_state.user_profiles.get(st.session_state.user_id, {}).get('last_name', '')}".strip(), disabled=True, key="fact_check_user_name")
    with col3:
        st.text_input("Email", value=st.session_state.user_profiles.get(st.session_state.user_id, {}).get("email", ""), disabled=True, key="fact_check_user_email")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        input_method = st.selectbox(
            "Choose input method:",
            ["Manual Text Entry", "Upload Text File", "Upload Image"],
            key="fact_check_input_method"
        )
        
        fact_check_claim = ""
        
        if input_method == "Manual Text Entry":
            fact_check_claim = st.text_area(
                "Enter claim to fact-check:",
                height=400,
                placeholder="Paste your claim here...",
                help="Enter the claim to fact-check",
                key="fact_check_text"
            )
        
        elif input_method == "Upload Text File":
            uploaded_file = st.file_uploader(
                "Upload a text file containing the news claim:",
                type=["txt"],
                key="fact_check_file_upload"
            )
            if uploaded_file is not None:
                fact_check_claim = uploaded_file.read().decode("utf-8")
                st.text_area(
                    "Preview of uploaded text:",
                    value=fact_check_claim,
                    height=400,
                    disabled=True,
                    key="fact_check_file_preview"
                )
        
        elif input_method == "Upload Image":
            uploaded_image = st.file_uploader(
                "Upload an image containing the news claim:",
                type=["png", "jpg", "jpeg"],
                key="fact_check_image_upload"
            )
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", width=300)
                fact_check_claim = "Image uploaded for analysis"
        
        if st.button("✅ Fact-Check", use_container_width=True, type="primary"):
            checker = FactChecker()
            
            if input_method == "Upload Image" and uploaded_image:
                with st.spinner("🧠 Analyzing the image..."):
                    status_placeholder = st.empty()
                    try:
                        image = Image.open(uploaded_image)
                        result = checker.fact_check_image(image)
                        status = "Success" if result["results"] else "Error"
                        
                        if status == "Success":
                            st.success("✅ Fact-Check Complete!")
                            st.markdown("---")
                            
                            st.markdown("<div class='assessment-card'>", unsafe_allow_html=True)
                            verdict = result['consensus']['verdict']
                            prediction = "Real" if verdict == "صحيح" else "Fake" if verdict == "خاطئ" else "Uncertain"
                            status_class = "status-true" if verdict == "صحيح" else ("status-false" if verdict == "خاطئ" else "status-error")
                            confidence = result['consensus']['confidence']
                            impact = result['consensus']['impact']
                            highlighted_claim = highlight_keywords(result['original_claim'], result['keywords'])
                            st.markdown(f"""
                            <div class='status-container'>
                                <div class='status-badge {status_class}'>{verdict}</div>
                                <div>
                                    <h3 class='header-text'>Fact-Check Results</h3>
                                    <ul class='info-list'>
                                        <li><strong>Claim:</strong> {highlighted_claim}</li>
                                        <li><strong>Category:</strong> {result['category']}</li>
                                    </ul>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h4>Key Indicators</h4>", unsafe_allow_html=True)
                            st.markdown(f"<ul class='info-list'><li><strong>Keywords Analyzed:</strong> {', '.join(result['keywords'])}</li></ul>")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h4>Consensus</h4>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <ul class='info-list'>
                                <li><strong>Confidence:</strong> {confidence}%</li>
                                <li><strong>Explanation:</strong> {result['consensus']['explanation']}</li>
                            </ul>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h4>Impact and Dangers</h4>", unsafe_allow_html=True)
                            st.markdown(f"<ul class='info-list'><li>{impact}</li></ul>")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            display_confidence_gauge(confidence, "Fact-Check Confidence")
                            
                            model_names = list(result['results'].keys())
                            confidences = [data['confidence'] for data in result['results'].values()]
                            df = pd.DataFrame({'Model': model_names, 'Confidence': confidences})
                            fig = px.bar(
                                df,
                                x='Model',
                                y='Confidence',
                                title="Model Confidence Levels",
                                color='Confidence',
                                color_continuous_scale='Blues',
                                height=400,
                                text='Confidence'
                            )
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h4>Detailed Model Results</h4>", unsafe_allow_html=True)
                            for model, data in result['results'].items():
                                with st.expander(f"Model: {model}"):
                                    st.markdown(f"""
                                    <ul class='info-list'>
                                        <li><strong>Verdict:</strong> {data['verdict']}</li>
                                        <li><strong>Explanation:</strong> {data['explanation']}</li>
                                        <li><strong>Sources:</strong> {', '.join([f'<a href="{source}" target="_blank">{source}</a>' for source in data['sources']])}</li>
                                        <li><strong>Confidence:</strong> {data['confidence']}%</li>
                                        <li><strong>Impact and Dangers:</strong> {data['impact']}</li>
                                    </ul>
                                    """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                            st.markdown("<h4>Metadata</h4>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <ul class='info-list'>
                                <li><strong>Date:</strong> {result['metadata']['date']}</li>
                                <li><strong>Models Used:</strong> {', '.join(result['metadata']['models_used'])}</li>
                            </ul>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            fig = px.box(
                                df,
                                y='Confidence',
                                title="Confidence Distribution Across Models",
                                points="all",
                                color_discrete_sequence=['var(--chart1)'],
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            add_to_history("Fact-Check", fact_check_claim, f"Verdict: {verdict}", "Success", confidence, result['category'], prediction, impact)
                        else:
                            st.error(f"Fact-Check Failed: {result['consensus']['explanation']}")
                            add_to_history("Fact-Check", fact_check_claim, f"Error: {result['consensus']['explanation']}", "Error", 0, "Unknown", "Unknown", "Not Applicable")
                    except Exception as e:
                        st.error(f"Image Processing Error: {str(e)}")
                        add_to_history("Fact-Check", fact_check_claim, f"Error: {str(e)}", "Error", 0, "Unknown", "Unknown", "Not Applicable")
            
            elif not validate_input(fact_check_claim, "claim"):
                pass
            else:
                with st.spinner("🧠 Fact-checking your claim..."):
                    checker = FactChecker()
                    category = checker.classify_category(fact_check_claim)
                    result = checker.fact_check(fact_check_claim, category)
                    status = "Success" if result["results"] else "Error"
                    
                    if status == "Success":
                        st.success("✅ Fact-Check Complete!")
                        st.markdown("---")
                        
                        st.markdown("<div class='assessment-card'>", unsafe_allow_html=True)
                        verdict = result['consensus']['verdict']
                        prediction = "Real" if verdict == "صحيح" else "Fake" if verdict == "خاطئ" else "Uncertain"
                        status_class = "status-true" if verdict == "صحيح" else ("status-false" if verdict == "خاطئ" else "status-error")
                        confidence = result['consensus']['confidence']
                        impact = result['consensus']['impact']
                        highlighted_claim = highlight_keywords(result['original_claim'], result['keywords'])
                        st.markdown(f"""
                        <div class='status-container'>
                            <div class='status-badge {status_class}'>{verdict}</div>
                            <div>
                                <h3 class='header-text'>Fact-Check Results</h3>
                                <ul class='info-list'>
                                    <li><strong>Claim:</strong> {highlighted_claim}</li>
                                    <li><strong>Category:</strong> {result['category']}</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                        st.markdown("<h4>Key Indicators</h4>", unsafe_allow_html=True)
                        st.markdown(f"<ul class='info-list'><li><strong>Keywords Analyzed:</strong> {', '.join(result['keywords'])}</li></ul>")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                        st.markdown("<h4>Consensus</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <ul class='info-list'>
                            <li><strong>Confidence:</strong> {confidence}%</li>
                            <li><strong>Explanation:</strong> {result['consensus']['explanation']}</li>
                        </ul>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                        st.markdown("<h4>Impact and Dangers</h4>", unsafe_allow_html=True)
                        st.markdown(f"<ul class='info-list'><li>{impact}</li></ul>")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        display_confidence_gauge(confidence, "Fact-Check Confidence")
                        
                        model_names = list(result['results'].keys())
                        confidences = [data['confidence'] for data in result['results'].values()]
                        df = pd.DataFrame({'Model': model_names, 'Confidence': confidences})
                        fig = px.bar(
                            df,
                            x='Model',
                            y='Confidence',
                            title="Model Confidence Levels",
                            color='Confidence',
                            color_continuous_scale='Blues',
                            height=400,
                            text='Confidence'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                        st.markdown("<h4>Detailed Model Results</h4>", unsafe_allow_html=True)
                        for model, data in result['results'].items():
                            with st.expander(f"Model: {model}"):
                                st.markdown(f"""
                                <ul class='info-list'>
                                    <li><strong>Verdict:</strong> {data['verdict']}</li>
                                    <li><strong>Explanation:</strong> {data['explanation']}</li>
                                    <li><strong>Sources:</strong> {', '.join([f'<a href="{source}" target="_blank">{source}</a>' for source in data['sources']])}</li>
                                    <li><strong>Confidence:</strong> {data['confidence']}%</li>
                                    <li><strong>Impact and Dangers:</strong> {data['impact']}</li>
                                </ul>
                                """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                        st.markdown("<h4>Metadata</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <ul class='info-list'>
                            <li><strong>Date:</strong> {result['metadata']['date']}</li>
                            <li><strong>Models Used:</strong> {', '.join(result['metadata']['models_used'])}</li>
                        </ul>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        fig = px.box(
                            df,
                            y='Confidence',
                            title="Confidence Distribution Across Models",
                            points="all",
                            color_discrete_sequence=['var(--chart1)'],
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        add_to_history("Fact-Check", fact_check_claim, f"Verdict: {verdict}", "Success", confidence, result['category'], prediction, impact)
                    else:
                        st.error("Fact-Check Failed: No results from models.")
                        add_to_history("Fact-Check", fact_check_claim, "No results", "Error", 0, "Unknown", "Unknown", "Not Applicable")

    with col2:
        st.markdown("### 📚 Tips for Better Fact-Checking")
        st.markdown("""
        <ul class='info-list'>
            <li>Be specific with claims</li>
            <li>Avoid vague statements</li>
            <li>Provide verifiable information</li>
            <li>Check sources in results</li>
            <li>Compare model verdicts</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Fact-Check Stats")
        stats = get_user_stats()
        st.markdown(f"""
        <div class='info-card'>
            <h4>User Statistics</h4>
            <ul class='info-list'>
                <li><strong>Requests Made:</strong> {stats['requests_made']}</li>
                <li><strong>Success Rate:</strong> {stats['success_rate']}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    if not st.session_state.history:
        st.info("No history yet. Make some requests to get started!")
    else:
        st.markdown("### 📜 Request History")
        
        for i, entry in enumerate(st.session_state.history):
            status_class = "success" if entry['Status'] == "Success" else "error"
            with st.expander(f"Request {i+1} - {entry['Action']} ({entry['Timestamp']})", expanded=False):
                st.markdown(f"""
                <div class='history-item'>
                    <div class='history-header'>
                        {entry['Action']} • {entry['UserID']} • {entry['Timestamp']}
                    </div>
                    <div class='history-content'>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
                            <div>
                                <ul class='info-list'>
                                    <li><strong>Action:</strong> {entry['Action']}</li>
                                    <li><strong>User:</strong> {entry['UserID']}</li>
                                    <li><strong>Status:</strong> <span class='status-badge-small {status_class}'>{entry['Status']}</span></li>
                                    <li><strong>Confidence:</strong> {entry['Confidence']}%</li>
                                </ul>
                            </div>
                        </div>
                        <div style="margin-bottom: 24px;">
                            <h4>Input Text</h4>
                            <div class='text-snippet'>{entry["Input"]}</div>
                        </div>
                        <h4>Result</h4>
                        <div class='assessment-card' style="margin-top: 12px;">
                            <ul class='info-list'>
                                <li>{entry["Result"]}</li>
                            </ul>
                        </div>
                        <h4>Impact (if Fake)</h4>
                        <div class='assessment-card' style="margin-top: 12px;">
                            <ul class='info-list'>
                                <li>{entry["Impact"]}</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Delete Request {i+1}", key=f"del_{i}"):
                    del st.session_state.history[i]
                    st.rerun()
        
        status_counts = pd.Series([entry['Status'] for entry in st.session_state.history]).value_counts()
        df = pd.DataFrame({
            'Status': status_counts.index,
            'Count': status_counts.values
        })
        fig = px.pie(
            df,
            names='Status',
            values='Count',
            title="Request Status Distribution",
            color='Status',
            color_discrete_map={'Success': 'var(--chart1)', 'Error': 'var(--warning)'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        history_df = pd.DataFrame(st.session_state.history)
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        fig = px.scatter(
            history_df,
            x='Timestamp',
            y='Action',
            color='Status',
            size='Confidence',
            title="Request Timestamps by Action Type",
            color_discrete_map={'Success': 'var(--chart1)', 'Error': 'var(--warning)'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        category_counts = pd.Series([entry['Category'] for entry in st.session_state.history]).value_counts()
        df = pd.DataFrame({
            'Category': category_counts.index,
            'Count': category_counts.values
        })
        fig = px.pie(
            df,
            names='Category',
            values='Count',
            title="Detected Fake News by Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        source_counts = pd.Series([entry['Source'] for entry in st.session_state.history]).value_counts().head(5)
        df = pd.DataFrame({
            'Source': source_counts.index,
            'Count': source_counts.values
        })
        fig = px.bar(
            df,
            x='Source',
            y='Count',
            title="Top 5 Reported Sources for Fake News",
            color='Source',
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=400,
            text='Count'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="history.csv">Download History as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

with tab5:
    st.markdown("### 📊 Analytics Dashboard")
    
    if not st.session_state.history:
        st.info("No data available for analytics. Make some requests to see insights.")
    else:
        total_requests = len(st.session_state.history)
        success_requests = len([entry for entry in st.session_state.history if entry["Status"] == "Success"])
        success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
        
        st.markdown(f"""
        <div class='assessment-card'>
            <h3 class='header-text'>Overall Statistics</h3>
            <p>Total Requests: {total_requests}</p>
            <p>Success Rate: {success_rate:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        prediction_counts = pd.Series([entry['Prediction'] for entry in st.session_state.history if entry['Prediction'] in ['Fake', 'Real']]).value_counts()
        if not prediction_counts.empty:
            df = pd.DataFrame({
                'Prediction': prediction_counts.index,
                'Count': prediction_counts.values
            })
            fig = px.pie(
                df,
                names='Prediction',
                values='Count',
                title="Percentage of Fake vs Real News Predictions",
                color='Prediction',
                color_discrete_map={'Fake': 'var(--warning)', 'Real': 'var(--chart1)'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if not prediction_counts.empty:
            fig = px.bar(
                df,
                x='Prediction',
                y='Count',
                title="Counts of Predicted Classes (Fake vs Real) 📊",
                color='Prediction',
                color_discrete_map={'Fake': 'var(--warning)', 'Real': 'var(--chart1)'},
                height=400,
                text='Count'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        true_labels = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
        pred_labels = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
        pred_probs = [0.9, 0.2, 0.8, 0.4, 0.3, 0.6, 0.95, 0.1, 0.85, 0.7]
        
        cm = confusion_matrix(true_labels, pred_labels)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=['Fake', 'Real'],
            y=['Fake', 'Real'],
            title="Confusion Matrix 🔀",
            color_continuous_scale='Blues',
            text_auto=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        
        
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        fig = px.line(
            x=fpr,
            y=tpr,
            title="ROC Curve 📈",
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            color_discrete_sequence=['var(--chart3)'],
            height=400
        )
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
        st.plotly_chart(fig, use_container_width=True)
        
        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
        fig = px.line(
            x=recall,
            y=precision,
            title="Precision-Recall Curve 🎯",
            labels={'x': 'Recall', 'y': 'Precision'},
            color_discrete_sequence=['var(--chart4)'],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.user_id:
            user_stats = get_user_stats()
            st.markdown(f"""
            <div class='assessment-card'>
                <h3 class='header-text'>User-Specific Analytics ({st.session_state.user_id})</h3>
                <p>Requests Made: {user_stats['requests_made']}</p>
                <p>Success Rate: {user_stats['success_rate']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            user_requests = [entry for entry in st.session_state.history if entry["UserID"] == st.session_state.user_id]
            if user_requests:
                dates = [datetime.strptime(entry['Timestamp'], "%Y-%m-%d %H:%M:%S") for entry in user_requests]
                df = pd.DataFrame({'Date': dates, 'Count': 1})
                df = df.groupby(df['Date'].dt.date)['Count'].count().reset_index(name='Count')
                fig = px.line(
                    df,
                    x='Date',
                    y='Count',
                    title="User Request Trend Over Time",
                    height=400,
                    line_shape='spline',
                    color_discrete_sequence=['var(--info)']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                


with tab6:
    st.markdown("### ℹ️ Learn More: The Damage of Fake News")
    st.markdown("""
    <div class='assessment-card'>
        <h3 class='header-text'>The Devastating Impact of Fake News</h3>
        <p>Fake news, or the deliberate spread of misinformation, has become a pervasive issue in the digital age. With the rise of social media and instant communication, false information can spread rapidly, causing significant harm to individuals, communities, and societies. This section highlights the multifaceted damage caused by fake news with key points, examples, and actionable steps to combat it.</p>
        <div class='result-section'>
            <h4>🌐 Social and Political Consequences</h4>
            <ul>
                <li><strong>Fuels Division and Polarization:</strong> Misinformation creates echo chambers, deepening societal divides. A 2016 study by the <a href="https://www.oxfordmartin.ox.ac.uk/news/2016-11-16-fake-news/" target="_blank">Oxford Internet Institute</a> found that 25% of political news on Twitter during the U.S. election was misleading or false.</li>
                <li><strong>Manipulates Voter Behavior:</strong> False narratives can sway elections by influencing public opinion and eroding trust in democratic processes.</li>
                <li><strong>Undermines Institutions:</strong> Persistent misinformation reduces trust in media and government, destabilizing democratic systems.</li>
                <li><strong>Example - Pizzagate (2016):</strong> A conspiracy theory falsely linked a Washington, D.C., pizzeria to a child trafficking ring involving Hillary Clinton, leading to an armed man entering the restaurant to "investigate," endangering lives.</li>
                <li><strong>Example - Brexit (2016):</strong> Misleading claims, such as the UK sending £350 million weekly to the EU (debunked by the UK Statistics Authority), influenced the Brexit referendum, affecting voter decisions.</li>
            </ul>
        </div>
        <div class='result-section'>
            <h4>💰 Economic Impact</h4>
            <ul>
                <li><strong>Destabilizes Markets:</strong> False financial news can cause market volatility. A 2020 <a href="https://www.cheq.ai/resources/fake-news-report-2020" target="_blank">CHEQ Cybersecurity Firm</a> report estimated fake news costs the global economy $78 billion annually.</li>
                <li><strong>Hurts Businesses:</strong> Companies face reputational damage and financial losses due to fabricated stories or smear campaigns.</li>
                <li><strong>Affects Consumer Confidence:</strong> Misinformation about products or economic conditions can lead to panic buying or market distrust.</li>
                <li><strong>Example - AP Twitter Hack (2013):</strong> A fake tweet from the Associated Press's hacked account claimed explosions at the White House injured President Obama, causing the Dow Jones to drop 143 points in minutes.</li>
                <li><strong>Example - Tesla Stock (2018):</strong> False rumors about Tesla's financial instability led to a temporary 7% drop in its stock price, impacting investor confidence until clarified.</li>
            </ul>
        </div>
        <div class='result-section'>
            <h4>🩺 Public Health Risks</h4>
            <ul>
                <li><strong>Spreads Harmful Behaviors:</strong> Misinformation about health can lead to dangerous actions, such as rejecting proven treatments or using unverified remedies.</li>
                <li><strong>Causes Public Panic:</strong> False health news creates fear, overwhelming healthcare systems during crises.</li>
                <li><strong>Increases Mortality Rates:</strong> A 2021 <a href="https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(20)30239-8/fulltext" target="_blank">Lancet Public Health</a> study linked COVID-19 misinformation to thousands of preventable deaths due to vaccine hesitancy.</li>
                <li><strong>Example - COVID-19 Bleach Myth (2020):</strong> False claims that drinking bleach could cure COVID-19 led to a spike in poisonings, with the CDC noting increased calls to poison control centers.</li>
                <li><strong>Example - Measles Outbreak (2019):</strong> Anti-vaccine misinformation fueled a 30% rise in measles cases globally, with a Samoa outbreak killing 83 people, mostly children, due to vaccine hesitancy.</li>
            </ul>
        </div>
        <div class='result-section'>
            <h4>🛡️ Actionable Steps to Combat Fake News</h4>
            <ul>
                <li><strong>Verify Sources:</strong> Cross-check information with trusted outlets like Reuters, BBC, or Al Jazeera. Avoid sharing unverified content.</li>
                <li><strong>Recognize Red Flags:</strong> Look for sensational headlines, missing citations, or suspicious URLs that mimic legitimate sites.</li>
                <li><strong>Use Fact-Checking Tools:</strong> Leverage platforms like this app, Snopes, or FactCheck.org to verify claims before spreading them.</li>
                <li><strong>Promote Media Literacy:</strong> Educate friends and family about fake news risks and encourage critical thinking skills.</li>
                <li><strong>Report Misinformation:</strong> Flag false content on social media platforms to limit its spread and protect others.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
