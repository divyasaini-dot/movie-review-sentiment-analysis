import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# English stopwords
stop_words = set(stopwords.words("english"))

# Keep negation and important sentiment words
important_words = {
    "not",
    "no",
    "nor",
    "never",
    "very",
    "too",
    "really",
    "quite"
}

# Remove important words from stopwords
stop_words = stop_words - important_words


# Contractions
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",

    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",

    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",

    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",

    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would"
}


def preprocess_text(text):

    # Handle missing values
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Normalize extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize using whitespace
    words = text.split()

    processed_words = []

    negation_words = {
        "not",
        "no",
        "nor",
        "never"
    }

    i = 0

    while i < len(words):

        word = words[i]

        # Combine negation with the following meaningful word
        if word in negation_words and i + 1 < len(words):

            next_word = words[i + 1]

            if next_word not in stop_words:

                processed_words.append(
                    word + "_" + next_word
                )

                i += 2
                continue

        # Remove normal stopwords
        if word not in stop_words:
            processed_words.append(word)

        i += 1

    return " ".join(processed_words)