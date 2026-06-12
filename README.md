Sanchez-O'Mullony Martínez, María Isabel, 123456

# PoS-Tagger for Spanish Language

## Project description
This project implements an HMM-based Part-of-Speech tagger for Spanish. The application uses a small manual Spanish corpus with word-level PoS annotations and trains an HMM PoS-tagger with NLTK. The project includes corpus creation, training, evaluation, and a command-line interface for custom sentence tagging.

## Prerequisites

They can be found in the [requirements](requirements.txt).

- Python >= 3.10
- `nltk` Python package
- `conllu` to get the corpus

## Installation
1. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage
Train the tagger and evaluate it on the built-in corpus:
```bash
python tagger.py
```

Tag a custom Spanish sentence:
```bash
python tagger.py --text "El perro corre rápido."
```

Show example corpus sentences together with English translations:
```bash
python tagger.py --show-translations
```

## Alternative Language
The chosen alternative language is Spanish.

- **What is a sentence?**
  A sentence is a sequence of tokens that represents a complete idea in Spanish. It begins with a capital letter or opening punctuation and ends with a full stop, question mark, or exclamation mark.

- **What is a word token?**
  A word token is an individual lexical unit such as "niño", "come", or "Madrid". Punctuation marks like "." and "?" are also represented as separate tokens.

- **Used PoS-tags with examples**
  This project uses the Universal Part-of-Speech tagset from NLTK. Example tags and Spanish examples:
  - `ADJ`: `grande`, `interesante`
  - `ADP`: `en`, `a`, `con`, `por`, `sobre`
  - `ADV`: `rápido`, `siempre`, `muy`
  - `AUX`: `está`, `son`, `es`, `puedes`
  - `CONJ`: `y`, `o`, `pero`
  - `DET`: `el`, `la`, `los`, `una`, `mi`
  - `NOUN`: `gato`, `libro`, `calor`, `museos`
  - `NUM`: `dos`, `tres`, `ocho`, `diez`
  - `PART`: `no`
  - `PRON`: `ella`, `nosotros`, `tú`
  - `PROPN`: `Madrid`, `Juan`, `Barcelona`
  - `PUNCT`: `.`, `,`, `?`
  - `VERB`: `come`, `vive`, `juega`, `quiero`

## Used AI
ChatGPT was used to find the best PoS Tagged Spanish Corpus.

- Can you find a PoS tagged corpus for spanish and that can be used in python?

## Implementation of the Requests
- The project trains and evaluates a Hidden Markov Model PoS-tagger in `tagger.py`. It uses NLTK's `HiddenMarkovModelTrainer` with Lidstone smoothing to handle the small Spanish corpus.
- The Spanish corpus is stored in `spanish_corpus.py` as a list of token/tag tuples.
- The corpus is split into training and test sentences inside `tagger.py`.
- The application computes accuracy on the test set and prints evaluation results.
- The tagger reads another input sentence with the `--text` option and outputs PoS tags as a list of tuples.
- The README describes the corpus, Spanish language details, tokens, sentences, and PoS-tag definitions.

## Corpus translation sample
A few corpus sentences with English translations:
- "Mi hermano come manzanas." — "My brother eats apples."
- "Ella vive en Madrid." — "She lives in Madrid."
- "El gato duerme en la silla." — "The cat sleeps on the chair."
- "No me gusta el café." — "I do not like coffee."
- "La ciudad tiene muchos museos." — "The city has many museums."
