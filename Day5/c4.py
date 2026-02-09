import os
import sys
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import NLP libraries
import spacy
from nltk.stem import SnowballStemmer

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# Initialize stemmer
stemmer = SnowballStemmer(language='english')

# Read file name
filename = input("Enter text file name for full text processing: ")

# File existence check
file_path = os.path.join(sys.path[0], filename)
if not os.path.exists(file_path):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

# Read file content
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# -------------------- OUTPUT 1 --------------------
print("\nOriginal Text Sample:")
print(content[:300])
print()

# -------------------- OUTPUT 2 --------------------
sample_words = "friendship studied was am is organizing matches"
doc_sample = nlp(sample_words)

print("=== Lemmatization: Individual Words ===")
for token in doc_sample:
    if not token.is_space:
        print(f"{token.text} -> {token.lemma_}")
print()

# -------------------- OUTPUT 3 --------------------
print("=== Stemming: Individual Words ===")
for word in sample_words.split():
    print(f"{word} --> {stemmer.stem(word)}")
print()

# Tokenize full text
doc_full = nlp(content)

# -------------------- OUTPUT 4 --------------------
print("=== Lemmatization: Full Text ===")
count = 0
for token in doc_full:
    if not token.is_space:
        print(f"{token.text} --> {token.lemma_}")
        count += 1
    if count == 50:
        break
print()

# -------------------- OUTPUT 5 --------------------
print("=== Stemming: Full Text ===")
count = 0
for token in doc_full:
    if not token.is_space:
        print(f"{token.text} --> {stemmer.stem(token.text.lower())}")
        count += 1
    if count == 50:
        break
print()

# -------------------- OUTPUT 6 --------------------
practice_words = "running good universities flies fairer is"
doc_practice = nlp(practice_words)

print("=== Practice 6.2: Lemmatization vs Stemming ===")
print("Word\t\tLemma\t\tStem")
print("------------------------------------------")

for token in doc_practice:
    stem = stemmer.stem(token.text.lower())
    print(f"{token.text}\t\t{token.lemma_}\t\t{stem}")
print()

# -------------------- OUTPUT 7 --------------------
print("Conclusion:")
print(
    "Lemmatization produces dictionary-based meaningful root words, while stemming may distort words "
    "by chopping suffixes. For NLP tasks like search, topic modeling, and information retrieval, "
    "lemmatization gives better and cleaner output."
)
