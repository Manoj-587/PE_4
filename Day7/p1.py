import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.simplefilter(action='ignore')

import spacy
from spacy.matcher import PhraseMatcher


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
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Install it using: python -m spacy download en_core_web_sm")
        sys.exit(1)

    # ============================
    # Step 3: Create Doc object
    # ============================
    doc = nlp(content)

    # ==================================================
    # Step 4.1: Rule Based Matching – Athlete Names
    # ==================================================
    athlete_matcher = PhraseMatcher(nlp.vocab)

    athlete_names = [
        "Sarah Claxton",
        "Sonia O'Sullivan",
        "Irina Shevchenko"
    ]

    athlete_patterns = [nlp.make_doc(name) for name in athlete_names]
    athlete_matcher.add("ATHLETE_NAMES", athlete_patterns)

    athlete_matches = athlete_matcher(doc)

    print("=== Matched Athlete Names ===")
    if athlete_matches:
        for match_id, start, end in athlete_matches:
            span = doc[start:end]
            print(f"- {span.text}")
    else:
        print("No athlete names found.")
    print()

    # ==================================================
    # Step 4.2: Rule Based Matching – Sports Events
    # ==================================================
    event_matcher = PhraseMatcher(nlp.vocab)

    event_names = [
        "European Indoor Championships",
        "World Cross Country Championships",
        "London marathon",
        "Bupa Great Ireland Run"
    ]

    event_patterns = [nlp.make_doc(event) for event in event_names]
    event_matcher.add("SPORT_EVENTS", event_patterns)

    event_matches = event_matcher(doc)

    print("=== Matched Sports Events ===")
    if event_matches:
        for match_id, start, end in event_matches:
            span = doc[start:end]
            print(f"- {span.text}")
    else:
        print("No sports events found.")
    print()


if __name__ == "__main__":
    main()
