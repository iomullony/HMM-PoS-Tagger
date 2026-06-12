Sanchez-O'Mullony Martínez, María Isabel, 123456

# PoS-Tagger for Spanish Language

## Project description

This project implements a Hidden Markov Model (HMM) Part-of-Speech (PoS) tagger for Spanish, an alternative language to English and German. The tagger is built using NLTK's `HiddenMarkovModelTrainer` with Lidstone smoothing to handle unseen tokens. It supports two corpus modes: a hand-crafted Spanish corpus of 40 annotated sentences (`spanish_corpus.py`), and the large-scale UD Spanish AnCora corpus in CoNLL-U format (`--use-ancora` flag). The project includes corpus creation, training, evaluation, and a command-line interface for tagging custom sentences.

## Prerequisites

- Python >= 3.10
- `nltk` Python package
- `conllu` Python package (required for parsing the AnCora corpus)

All dependencies are listed in [requirements.txt](requirements.txt) and can be installed with `pip install -r requirements.txt`.

## Installation

1. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* To use the UD Spanish AnCora corpus for better accuracy, download `es_ancora-ud-train.conllu` from [Universal Dependencies](https://universaldependencies.org/) and place it in the project root directory. This step is not required — the tagger works out of the box with the built-in manual corpus.

## Basic Usage

**Train and evaluate the tagger on the built-in manual corpus:**
```bash
python tagger.py
```

**Tag a custom Spanish sentence:**
```bash
python tagger.py --text "El perro corre rápido."
```

**Show corpus sentences with their English translations:**
```bash
python tagger.py --show-translations
```

**Use the larger UD AnCora corpus (requires the optional download above):**
```bash
python tagger.py --use-ancora
```

**Combine flags — tag a custom sentence using the AnCora-trained model:**
```bash
python tagger.py --use-ancora --text "¿Cómo estás hoy?"
```

### Sample output

```
$ python tagger.py --text "El perro corre rápido."

Evaluation results
-------------------
Train sentences: 32
Test sentences:  8
Accuracy:       76.60%

Test sentence tagging samples:
- original: El tren llega a las diez .
  tagged:   [('El', 'DET'), ('tren', 'NOUN'), ('llega', 'PUNCT'), ('a', 'ADP'), ('las', 'DET'), ('diez', 'NOUN'), ('.', 'PUNCT')]
- original: Hace frío en invierno .
  tagged:   [('Hace', 'PRON'), ('frío', 'VERB'), ('en', 'ADP'), ('invierno', 'PROPN'), ('.', 'PUNCT')]
- original: Puedo leer y escribir en español .
  tagged:   [('Puedo', 'DET'), ('leer', 'NOUN'), ('y', 'PUNCT'), ('escribir', 'AUX'), ('en', 'ADP'), ('español', 'NOUN'), ('.', 'PUNCT')]
- original: Tú tienes una bicicleta nueva .
  tagged:   [('Tú', 'PRON'), ('tienes', 'VERB'), ('una', 'DET'), ('bicicleta', 'NOUN'), ('nueva', 'ADJ'), ('.', 'PUNCT')]
- original: El equipo juega bien .
  tagged:   [('El', 'DET'), ('equipo', 'NOUN'), ('juega', 'VERB'), ('bien', 'ADJ'), ('.', 'PUNCT')]

Custom tagging:
Input sentence: El perro corre rápido.
Tokens: ['El', 'perro', 'corre', 'rápido', '.']
Tagged output: [('El', 'DET'), ('perro', 'NOUN'), ('corre', 'VERB'), ('rápido', 'ADV'), ('.', 'PUNCT')]
```

> **Note:** Accuracy on the manual corpus is limited by the small training set (32 sentences). Using `--use-ancora` significantly improves results as AnCora contains thousands of annotated sentences.

## Alternative Language

The chosen alternative language is **Spanish** (Español), a Romance language descended from Latin. It is the world's second most spoken native language and is the official language of 20 countries. Spanish uses the Latin alphabet with the addition of `ñ` and accented vowels (`á`, `é`, `í`, `ó`, `ú`). See [Wikipedia: Spanish language](https://en.wikipedia.org/wiki/Spanish_language).

### What is a sentence?

A sentence (*oración*) in Spanish is a sequence of tokens that expresses a complete idea. It begins with a capital letter — or an opening punctuation mark (`¿`, `¡`) in the case of questions and exclamations — and ends with a full stop (`.`), question mark (`?`), or exclamation mark (`!`). Spanish is notable for its inverted opening punctuation, which has no equivalent in English.

### What is a word token?

A word token (*token*) is an individual lexical unit obtained by splitting a sentence on whitespace after separating punctuation. Examples: `niño`, `come`, `Madrid`. Punctuation marks such as `.`, `,`, `?`, `¿` are also represented as separate tokens. Contractions such as *al* (a + el) and *del* (de + el) are treated as single tokens and tagged as `ADP`.

### Used PoS-tags with examples

This project uses the **Universal Part-of-Speech tagset** as described in the NLTK Book, Chapter 5 (see https://www.nltk.org/book/ch05.html). Spanish examples are given for each tag:

| Tag | Description | Spanish examples |
|-----|-------------|-----------------|
| `ADJ` | Adjective | `grande`, `interesante`, `nueva`, `difícil` |
| `ADP` | Adposition (preposition) | `en`, `a`, `con`, `por`, `sobre`, `al`, `del` |
| `ADV` | Adverb | `rápido`, `siempre`, `muy`, `hoy`, `mañana`, `ayer` |
| `AUX` | Auxiliary verb | `está`, `son`, `es`, `puedes`, `están` |
| `CONJ` | Coordinating conjunction | `y`, `o`, `pero` |
| `DET` | Determiner | `el`, `la`, `los`, `una`, `mi`, `su`, `este` |
| `NOUN` | Noun | `gato`, `libro`, `calor`, `museos`, `frío` |
| `NUM` | Numeral | `dos`, `tres`, `ocho`, `diez` |
| `PART` | Particle | `no` |
| `PRON` | Pronoun | `ella`, `nosotros`, `tú`, `me`, `se` |
| `PROPN` | Proper noun | `Madrid`, `Juan`, `Barcelona`, `María` |
| `PUNCT` | Punctuation | `.`, `,`, `?`, `¿`, `!` |
| `VERB` | Verb | `come`, `vive`, `juega`, `quiero`, `aprender` |
| `X` | Other / unknown | used for unclassified tokens (AnCora fallback) |

**Notes on Spanish-specific tagging decisions:**
- Temporal adverbs *hoy*, *mañana*, and *ayer* are tagged `ADV`, not `NOUN`, even though they can appear in subject position.
- The negation particle *no* is tagged `PART` following the Universal PoS tagset.
- Contractions *al* and *del* are tagged `ADP` as single tokens.
- *Frío* is tagged `NOUN` in *hace frío* (it is the object of the verb *hacer*) but `ADJ` in adjectival use (e.g., *agua fría*).

## Used AI

ChatGPT (GPT-4) was used in two parts of the project.

**1 — Finding a suitable PoS-tagged Spanish corpus:**

- *"Can you find a PoS-tagged corpus for Spanish that can be used in Python?"*
- *"Which Universal Dependencies treebanks are available for Spanish and how do I download them?"*

**2 — Verifying individual PoS-tag assignments in the manual corpus:**

- *"In the sentence 'Hoy hace mucho calor', what is the correct Universal PoS tag for 'Hoy'? Is it ADV or NOUN?"*
- *"Is 'frío' in the sentence 'Hace frío en invierno' a NOUN or an ADJ under the Universal PoS tagset?"*
- *"In 'Siempre estudio antes del examen', what PoS tag should 'antes' receive?"*

## Implementation of the Requests

The following maps each project requirement to its implementation in the code:

- **Runs from the command line (no GUI):** `tagger.py` uses `argparse` and is fully controlled via command-line flags (`--text`, `--use-ancora`, `--show-translations`).
- **Uses Python >= 3.10:** Confirmed; uses f-strings, `match`-compatible syntax, and type hints consistent with Python 3.10+.
- **No AI or online APIs:** All processing is local. NLTK and `conllu` are offline libraries; no external API calls are made.
- **Evaluate tagged corpus:** `print_evaluation()` in `tagger.py` computes and prints accuracy on the test split, along with tagged samples.
- **Split corpus into train and test sentences:** `split_corpus()` in `tagger.py` performs an 80/20 split on the tagged sentence list.
- **Create HMM PoS-tagger:** `train_hmm_tagger()` in `tagger.py` uses `nltk.tag.hmm.HiddenMarkovModelTrainer` with Lidstone smoothing (`LidstoneProbDist`, λ=0.1).
- **Show accuracy of the PoS-tagger:** Printed by `print_evaluation()` as a percentage alongside train/test sentence counts.
- **Read another text and output PoS-tags as a list of tuples:** `tag_text()` in `tagger.py` tokenizes a custom input sentence and returns the tagged output as a list of `(token, tag)` tuples, printed to stdout. Invoked via `--text`.
- **Corpus documentation:** The manual corpus is stored in `spanish_corpus.py` as three parallel lists: raw sentences, English translations, and tagged sentences (lists of `(word, tag)` tuples), following the NLTK convention.