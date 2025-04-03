import re
import pandas as pd
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer

# Initialize Farasa tools
segmenter = FarasaSegmenter()
stemmer = FarasaStemmer()

# Load dataset
path = './data_set/train_set.csv'
fake_data = pd.read_csv(path, header=0, encoding='utf-8')

if fake_data.empty:
    raise ValueError("The DataFrame is empty. Please check the CSV file ❌")
else:
    print("DataFrame loaded successfully ✅")

# Limit to first 100 rows
fake_data = fake_data.head(100)

# Define custom Arabic stopwords
# Load Arabic stopwords from a file
stopwords_path = './data_set/algerian_arabic_stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as file:
    ARABIC_STOPWORDS = set(file.read().splitlines())

# Define functions
def clean_text(text):
    """Basic text cleaning for Arabic"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    text = arabic_diacritics.sub('', text)  # Remove diacritics (tashkeel)
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alif
    text = re.sub(r'ة', 'ه', text)  # Normalize Teh Marbuta
    text = re.sub(r'ى', 'ي', text)  # Normalize Ya
    text = re.sub(r'[ؤئ]', 'ء', text)  # Normalize Hamzas
    text = re.sub(r'[^\u0600-\u06FF\s0-9]', '', text)  # Remove non-Arabic characters except spaces and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text

def tokenize_and_stem(text):
    """Tokenization, stemming, and stopword removal using Farasa"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    tokens = segmenter.segment(text).split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Remove stopwords
    final_tokens = [word for word in stemmed_tokens if word not in ARABIC_STOPWORDS]
    
    return " ".join(final_tokens)

# Ensure the expected column exists
if 'text' not in fake_data.columns:
    raise KeyError("The dataset does not contain a 'text' column.")

# Apply preprocessing with progress tracking
cleaned_texts = []
for index, row in fake_data.iterrows():
    cleaned_texts.append(tokenize_and_stem(clean_text(row['text'])))
    progress = ((index + 1) / 100) * 100
    print(f"Progress: {progress:.0f}%")

# Save preprocessed dataset
fake_data['cleaned_text'] = cleaned_texts
output_path = "preprocessed_100_rows.csv"
fake_data[['label', 'source', 'cleaned_text']].to_csv(output_path, index=False)

print("Processing of 100 rows completed! ✅")
