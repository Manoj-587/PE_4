import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from collections import Counter

def main():
    # ============================
    # Step 0: Input text filename
    # ============================
    filename = input("Enter text file name: ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load text file
    # ============================
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    # ============================
    # Step 2: Load spaCy model
    # ============================
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Apply NER
    # ============================
    doc = nlp(content)

    print("=== Named Entities (PERSON, GPE, DATE) ===")

    entity_counter = Counter()

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "DATE"]:
            print(f"{ent.text} ({ent.label_})")
            entity_counter[ent.label_] += 1

    print()

    # ============================
    # Step 4: Entity Frequency
    # ============================
    print("=== Entity Frequency ===")
    for entity, count in entity_counter.items():
        print(f"{entity}: {count}")

if __name__ == "__main__":
    main()
