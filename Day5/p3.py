import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore')

filename = input().strip()
filepath = os.path.join(sys.path[0], filename)

if not os.path.exists(filepath):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

with open(filepath, "r", encoding="utf-8") as file:
    content = file.read()

print("First 10 lines from the file:")
lines = content.splitlines()
for line in lines[:10]:
    print(line)
print()

doc = nlp(content)

tokens = [token.text for token in doc[:20]]
print("First 20 tokens:")
print(tokens)
print()

print("POS Tagging Output:")
print("Word\tPOS\tTag")
print("-" * 30)

for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.tag_}")
