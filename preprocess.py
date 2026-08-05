import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

negation_words = {"no", "not", "nor"}

stop_words = stop_words - negation_words

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and non-alphabetic words
    words = [
        word for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)