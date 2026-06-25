Sánchez-O'Mullony Martínez, María Isabel, 123456

# PoS-Tagger for Spanish Language

## Project description

This project implements a command-line Hidden Markov Model (HMM) Part-of-Speech (PoS) tagger for Spanish. PoS tagging means assigning a grammatical category to every token in a sentence, for example:

```python
[("El", "DET"), ("perro", "NOUN"), ("corre", "VERB"), (".", "PUNCT")]
```

The main program is [`tagger.py`](tagger.py). It trains an HMM using NLTK's `HiddenMarkovModelTrainer`, evaluates the trained tagger on a test set, and can tag a new Spanish sentence supplied by the user. The project has two corpus modes:

1. A manually created Spanish corpus of 40 tagged sentences in [`spanish_corpus.py`](spanish_corpus.py).
2. The larger Universal Dependencies Spanish AnCora corpus in CoNLL-U format, loaded from [`es_ancora-ud-train.conllu`](es_ancora-ud-train.conllu) when the `--use-ancora` flag is used.

The small manual corpus makes the data structure and grammatical decisions easy to inspect. The AnCora corpus demonstrates how much the same HMM model improves when trained on a larger real annotated corpus.

### HMM theory

A Hidden Markov Model is a probabilistic sequence model. In this project, the observed sequence is the sequence of word tokens, and the hidden sequence is the sequence of PoS tags. For example, in the sentence:

```text
El perro corre.
```

the observed tokens are:

```python
["El", "perro", "corre", "."]
```

and the hidden tags are:

```python
["DET", "NOUN", "VERB", "PUNCT"]
```

The model is called "hidden" because when a new sentence is given to the program, only the words are visible. The tag sequence must be inferred.

The HMM learns three main probability types:

- **Initial probabilities:** how likely a tag is at the beginning of a sentence, for example `P(DET)` or `P(PRON)`.
- **Transition probabilities:** how likely one tag is after another, for example `P(NOUN | DET)` or `P(VERB | PRON)`.
- **Emission probabilities:** how likely a word is for a given tag, for example `P(perro | NOUN)` or `P(corre | VERB)`.

The Markov assumption says that each tag depends mainly on the previous tag:

```text
P(t_i | t_1, ..., t_{i-1}) ≈ P(t_i | t_{i-1})
```

For a sentence with words `w_1 ... w_n`, the tagger searches for the tag sequence `t_1 ... t_n` that maximizes:

```text
P(t_1) * P(w_1 | t_1) *
P(t_2 | t_1) * P(w_2 | t_2) *
...
P(t_n | t_{n-1}) * P(w_n | t_n)
```

NLTK uses Viterbi decoding internally to find this best tag sequence efficiently. The project also uses Lidstone smoothing with gamma `0.1`:

```python
LidstoneProbDist(fdist, 0.1, bins)
```

Smoothing gives a small non-zero probability to unseen events. This is important because the manual corpus is small, so many words and tag transitions are not in the training data.

## Prerequisites

- Python >= 3.10 (Tested with Python 3.10.12)
- `nltk` Python package
- `conllu` Python package

All dependencies are listed in [`requirements.txt`](requirements.txt).

## Installation

1. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. The project already includes `es_ancora-ud-train.conllu`. If the file is missing, download the Universal Dependencies Spanish AnCora training file from [GitHub](https://github.com/UniversalDependencies/UD_Spanish-AnCora) and place it in the project root with this exact name:

   ```text
   es_ancora-ud-train.conllu
   ```

The program works without AnCora because it can use the built-in manual corpus.

## Basic Usage

Train and evaluate the tagger on the built-in manual corpus:

```bash
python tagger.py
```

Tag a custom Spanish sentence:

```bash
python tagger.py --text "El perro corre rápido."
```

Show example corpus sentences with English translations:

```bash
python tagger.py --show-translations
```

Use the larger UD AnCora corpus:

```bash
python tagger.py --use-ancora
```

Tag a custom sentence using the AnCora-trained model:

```bash
python tagger.py --use-ancora --text "¿Cómo estás hoy?"
```

### Sample output with manual corpus

```text
$ python tagger.py --text "El perro corre rápido."

Evaluation results
-------------------
Train sentences: 32
Test sentences:  8
Accuracy:       50.00%

Per-tag accuracy:
  Tag          Correct   Total  Accuracy
  ----------------------------------------
  ADJ                0       1     0.00%
  ADP                3       5    60.00%
  ADV                0       4     0.00%
  AUX                0       1     0.00%
  CCONJ              1       1   100.00%
  DET                3       3   100.00%
  NOUN               3       8    37.50%
  NUM                0       1     0.00%
  PRON               1       1   100.00%
  PROPN              1       3    33.33%
  PUNCT              8      10    80.00%
  VERB               3       8    37.50%

Test sentence tagging samples:
- original: Hoy hace mucho calor .
  tagged:   [('Hoy', 'DET'), ('hace', 'NOUN'), ('mucho', 'AUX'), ('calor', 'ADJ'), ('.', 'PUNCT')]
- original: Juan y María van al mercado .
  tagged:   [('Juan', 'PRON'), ('y', 'CCONJ'), ('María', 'NOUN'), ('van', 'VERB'), ('al', 'ADP'), ('mercado', 'PROPN'), ('.', 'PUNCT')]
- original: Tengo dos hermanos .
  tagged:   [('Tengo', 'DET'), ('dos', 'NOUN'), ('hermanos', 'ADJ'), ('.', 'PUNCT')]
- original: ¿ Puedes ayudarme , por favor ?
  tagged:   [('¿', 'DET'), ('Puedes', 'NOUN'), ('ayudarme', 'PUNCT'), (',', 'ADV'), ('por', 'AUX'), ('favor', 'ADJ'), ('?', 'PUNCT')]
- original: El niño juega con su pelota .
  tagged:   [('El', 'DET'), ('niño', 'NOUN'), ('juega', 'VERB'), ('con', 'ADP'), ('su', 'DET'), ('pelota', 'NOUN'), ('.', 'PUNCT')]

Custom tagging:
Input sentence: El perro corre rápido.
Tokens: ['El', 'perro', 'corre', 'rápido', '.']
Tagged output: [('El', 'DET'), ('perro', 'NOUN'), ('corre', 'AUX'), ('rápido', 'ADJ'), ('.', 'PUNCT')]
```

The 50% accuracy on the manual corpus is expected: with only 40 sentences and a fixed 80/20 split, the test set contains 8 sentences with many words unseen during training. The per-tag table shows that closed-class tags (DET, CCONJ, PUNCT) are tagged correctly while open-class tags (VERB, NOUN, ADV) suffer from data sparsity.

### Sample output with ancora corpus
```text
$ python tagger.py --use-ancora --text "El perro corre rápido."

Loaded 14287 UD AnCora sentences.
Evaluation results
-------------------
Train sentences: 11429
Test sentences:  2858
Accuracy:       94.98%

Per-tag accuracy:
  Tag          Correct   Total  Accuracy
  ----------------------------------------
  ADJ             5166    6038    85.56%
  ADP            14013   14033    99.86%
  ADV             2854    2993    95.36%
  AUX             2132    2191    97.31%
  CCONJ           2396    2403    99.71%
  DET            13548   13665    99.14%
  INTJ               5      27    18.52%
  NOUN           15344   16248    94.44%
  NUM             1300    1495    86.96%
  PRON            4901    5318    92.16%
  PROPN           5955    6681    89.13%
  PUNCT          10657   10658    99.99%
  SCONJ           1709    2026    84.35%
  SYM               77      91    84.62%
  VERB            6953    7737    89.87%
  X                  0       3     0.00%

Test sentence tagging samples:
- original: Sin embargo , los consumidores suecos miran con cierta incredulidad y sorpresa que estos acuerdos de precios existan entre las empresas de su país .
  tagged:   [('Sin', 'ADP'), ('embargo', 'NOUN'), (',', 'PUNCT'), ('los', 'DET'), ('consumidores', 'NOUN'), ('suecos', 'ADJ'), ('miran', 'VERB'), ('con', 'ADP'), ('cierta', 'DET'), ('incredulidad', 'NOUN'), ('y', 'CCONJ'), ('sorpresa', 'NOUN'), ('que', 'SCONJ'), ('estos', 'DET'), ('acuerdos', 'NOUN'), ('de', 'ADP'), ('precios', 'NOUN'), ('existan', 'VERB'), ('entre', 'ADP'), ('las', 'DET'), ('empresas', 'NOUN'), ('de', 'ADP'), ('su', 'DET'), ('país', 'NOUN'), ('.', 'PUNCT')]
- original: La titular de el Juzgado de Instrucción número 3 de Vic , Gladis López , cursó ayer las citaciones de los tres monitores y una orden a los Mossos d'Esquadra para que _ efectúen una reconstrucción de los hechos , también la próxima semana , según fuentes judiciales .
  tagged:   [('La', 'DET'), ('titular', 'NOUN'), ('de', 'ADP'), ('el', 'DET'), ('Juzgado', 'PROPN'), ('de', 'ADP'), ('Instrucción', 'DET'), ('número', 'NOUN'), ('3', 'NUM'), ('de', 'ADP'), ('Vic', 'PROPN'), (',', 'PUNCT'), ('Gladis', 'PROPN'), ('López', 'PROPN'), (',', 'PUNCT'), ('cursó', 'CCONJ'), ('ayer', 'ADV'), ('las', 'DET'), ('citaciones', 'NOUN'), ('de', 'ADP'), ('los', 'DET'), ('tres', 'NUM'), ('monitores', 'NOUN'), ('y', 'CCONJ'), ('una', 'DET'), ('orden', 'NOUN'), ('a', 'ADP'), ('los', 'DET'), ('Mossos', 'PROPN'), ("d'Esquadra", 'PROPN'), ('para', 'ADP'), ('que', 'SCONJ'), ('_', 'PRON'), ('efectúen', 'VERB'), ('una', 'DET'), ('reconstrucción', 'NOUN'), ('de', 'ADP'), ('los', 'DET'), ('hechos', 'NOUN'), (',', 'PUNCT'), ('también', 'ADV'), ('la', 'DET'), ('próxima', 'ADJ'), ('semana', 'NOUN'), (',', 'PUNCT'), ('según', 'ADP'), ('fuentes', 'NOUN'), ('judiciales', 'ADJ'), ('.', 'PUNCT')]
- original: Hacienda ya ha tramitado todas las devoluciones rápidas .
  tagged:   [('Hacienda', 'PROPN'), ('ya', 'ADV'), ('ha', 'AUX'), ('tramitado', 'VERB'), ('todas', 'DET'), ('las', 'DET'), ('devoluciones', 'NOUN'), ('rápidas', 'ADJ'), ('.', 'PUNCT')]
- original: Debido a ella , tienen , además , la facultad de aprender y reaccionar ante nuevas situaciones .
  tagged:   [('Debido', 'ADJ'), ('a', 'ADP'), ('ella', 'PRON'), (',', 'PUNCT'), ('tienen', 'VERB'), (',', 'PUNCT'), ('además', 'ADV'), (',', 'PUNCT'), ('la', 'DET'), ('facultad', 'NOUN'), ('de', 'ADP'), ('aprender', 'VERB'), ('y', 'CCONJ'), ('reaccionar', 'VERB'), ('ante', 'ADP'), ('nuevas', 'ADJ'), ('situaciones', 'NOUN'), ('.', 'PUNCT')]
- original: Los neonatos tienen que tener la oportunidad de convertir se , gracias a sus nombres , en mémoris ambulantes de los avatares de la Historia .
  tagged:   [('Los', 'DET'), ('neonatos', 'NOUN'), ('tienen', 'VERB'), ('que', 'PRON'), ('tener', 'VERB'), ('la', 'DET'), ('oportunidad', 'NOUN'), ('de', 'ADP'), ('convertir', 'VERB'), ('se', 'PRON'), (',', 'PUNCT'), ('gracias', 'NOUN'), ('a', 'ADP'), ('sus', 'DET'), ('nombres', 'NOUN'), (',', 'PUNCT'), ('en', 'ADP'), ('mémoris', 'DET'), ('ambulantes', 'NOUN'), ('de', 'ADP'), ('los', 'DET'), ('avatares', 'NOUN'), ('de', 'ADP'), ('la', 'DET'), ('Historia', 'PROPN'), ('.', 'PUNCT')]

Custom tagging:
Input sentence: El perro corre rápido.
Tokens: ['El', 'perro', 'corre', 'rápido', '.']
Tagged output: [('El', 'DET'), ('perro', 'NOUN'), ('corre', 'VERB'), ('rápido', 'ADJ'), ('.', 'PUNCT')]
```

`INTJ` at 18.52% and `X` at 0% can be seen. However, there are only 27 and 3 tokens respectively. With an HMM trained on frequency, these tags are so rare that the model almost never predicts them. It's not a real problem, just a small-data artifact.

### Evaluation results

With the manual corpus:

- Train sentences: 32
- Test sentences: 8
- Accuracy: 50.00%

With the AnCora corpus:

- Loaded sentences: 14,287
- Loaded tokens: 469,667
- Train sentences: 11,429
- Test sentences: 2,858
- Accuracy: 94.98%

The AnCora result is much better because the HMM sees more vocabulary, more tag transitions, and more realistic Spanish sentence structures.

## Alternative Language

The chosen alternative language is **Spanish** (*Español*), a Romance language descended from Latin. Spanish uses the Latin alphabet with `ñ` and accented vowels such as `á`, `é`, `í`, `ó`, and `ú`. It also uses inverted opening punctuation marks (`¿`, `¡`) for questions and exclamations.

Spanish grammar is relevant for PoS tagging because nouns have gender and number, determiners and adjectives often agree with nouns, and verb forms encode person, number, tense, mood, and aspect. Spanish word order is also flexible, so context is important for choosing the correct tag.

### What is a sentence?

A sentence is a sequence of tokens that expresses a complete idea. In Spanish, a written sentence normally begins with a capital letter or an opening punctuation mark such as `¿` or `¡`, and ends with `.`, `?`, or `!`.

In this project, a sentence is represented as a list of tagged tokens:

```python
[("La", "DET"), ("profesora", "NOUN"), ("explica", "VERB"), ("la", "DET"), ("lección", "NOUN"), (".", "PUNCT")]
```

### What is a word token?

A token is one unit that receives a PoS tag. Tokens can be words or punctuation marks. The tokenizer in [`tagger.py`](tagger.py) separates punctuation from words, so:

```text
¿Dónde está el baño?
```

becomes:

```python
["¿", "Dónde", "está", "el", "baño", "?"]
```

Contractions such as `al` (`a` + `el`) and `del` (`de` + `el`) are treated as single tokens in the manual corpus and tagged as `ADP`.

### Corpus

The manual corpus is stored in [`spanish_corpus.py`](spanish_corpus.py) as three parallel lists:

- `spanish_corpus_sentences`: 40 Spanish sentences.
- `spanish_corpus_translations`: English translations of the 40 sentences.
- `spanish_tagged_sentences`: the tagged corpus as lists of `(word, tag)` tuples.

The manual corpus contains:

- 40 sentences
- 238 tagged tokens
- 154 unique token forms
- 13 PoS tags

The optional AnCora corpus is stored in CoNLL-U format. The loader extracts the token form and Universal PoS tag from each token.

### Used PoS-tags with examples

This project uses the Universal Part-of-Speech tagset style described in NLTK Book, Chapter 5. Spanish examples are given for each tag:

| Tag | Description | Spanish examples |
| --- | --- | --- |
| `ADJ` | Adjective | `grande`, `interesante`, `nueva`, `difícil` |
| `ADP` | Adposition / preposition | `en`, `a`, `con`, `por`, `sobre`, `al`, `del` |
| `ADV` | Adverb | `rápido`, `siempre`, `muy`, `hoy`, `mañana`, `ayer` |
| `AUX` | Auxiliary verb | `está`, `son`, `es`, `puedes` |
| `CCONJ` | Coordinating conjunction | `y`, `o`, `pero` |
| `DET` | Determiner | `el`, `la`, `los`, `una`, `mi`, `su`, `este` |
| `NOUN` | Common noun | `gato`, `libro`, `calor`, `museos`, `frío` |
| `NUM` | Numeral | `dos`, `tres`, `ocho`, `diez` |
| `PART` | Particle | `no` |
| `PRON` | Pronoun | `ella`, `nosotros`, `tú`, `me`, `se` |
| `PROPN` | Proper noun | `Madrid`, `Juan`, `Barcelona`, `María` |
| `PUNCT` | Punctuation | `.`, `,`, `?`, `¿`, `!` |
| `VERB` | Main verb | `come`, `vive`, `juega`, `quiero`, `aprender` |
| `X` | Other / unknown | rare fallback category in UD-style data |

Spanish-specific tagging decisions:

- Temporal words such as `hoy`, `mañana`, and `ayer` are tagged as `ADV`.
- The negation word `no` is tagged as `PART` in the manual corpus.
- `al` and `del` are tagged as `ADP` in the manual corpus.
- `frío` is tagged as `NOUN` in `Hace frío en invierno`, because it functions as a noun in that expression.

## Used AI

ChatGPT and Claude were used as assistance in three areas: finding a suitable corpus, verifying individual tagging decisions, and reviewing the documentation. The program itself does not call any AI or online API.

Prompts used:

- "Can you find a PoS-tagged corpus for Spanish that can be used in Python?" — to identify available tagged corpora and discover the UD AnCora treebank.
- "Which Universal Dependencies treebanks are available for Spanish and how do I download them?" — to find the correct CoNLL-U file and download instructions.
- "Can you give me 40 simple sentences in Spanish?" — to create the Spanish corpus.
- "For these sentences, can you give the translation in English?" — to get the translated sentences in Spanish and English.
- "Can you give the tags for each sentence" — to have the tagged corpus.
- "Is it better to randomize the split of data even though I get worse accuracy in my manual corpus?" — to understand whether shuffling before splitting improves evaluation reliability.
- "Can you check for grammar mistakes and bad written phrases in this report?" — to rewrite the README documentation in a better way.

## Implementation of the Requests

The following list maps the project requirements to the implementation in the code.

- **Runs from the command line:** [`tagger.py`](tagger.py) uses `argparse` and is controlled by command-line flags such as `--text`, `--use-ancora`, and `--show-translations`, without using a GUI.
- **Uses Python >= 3.10:** The project was tested with Python 3.10.12.
- **No AI or online APIs in the application:** The tagger runs locally. NLTK and `conllu` are offline Python libraries, and the program does not call ChatGPT, Codex, or any web API.
- **Find or create a tagged corpus:** The project includes a manual Spanish tagged corpus in [`spanish_corpus.py`](spanish_corpus.py) and can optionally use the larger UD Spanish AnCora corpus from [`es_ancora-ud-train.conllu`](es_ancora-ud-train.conllu).
- **Use the NLTK list-of-tuples data structure:** The manual corpus stores each sentence as a list of `(word, tag)` tuples, for example `[("El", "DET"), ("perro", "NOUN")]`.
- **Provide English translations:** The manual corpus translations are stored in `spanish_corpus_translations` and can be displayed with `python tagger.py --show-translations`.
- **Document sentences, tokens, and PoS tags:** The [Alternative Language](#alternative-language) section explains Spanish sentences, word tokens, the corpus, and the used PoS tags with examples.
- **Split corpus into train and test sentences:** `split_corpus()` performs a reproducible 80/20 split using `TRAIN_RATIO = 0.8` and a fixed `RANDOM_SEED = 42`, ensuring identical train and test sets across runs.
- **Load AnCora corpus:** `load_ancora_tagged_sentences()` parses the CoNLL-U file using the `conllu` library, skipping multiword token spans and entries where `upos` is absent or `"_"`, to avoid polluting the training data with untagged surface tokens.
- **Create an HMM PoS tagger:** `train_hmm_tagger()` uses `nltk.tag.hmm.HiddenMarkovModelTrainer` with supervised learning and Lidstone smoothing (`LidstoneProbDist(fdist, 0.1, bins)`) to handle unseen word-tag pairs that do not appear in the small training set.
- **Evaluate the tagged corpus and show accuracy:** `print_evaluation()` evaluates the trained tagger on the test set and prints the number of train and test sentences together with the overall accuracy percentage.
- **Detailed per-tag evaluation:** `print_detailed_evaluation()` computes per-tag accuracy over the test set, showing which PoS categories the model handles well (e.g. `ADP`, `DET`, `PUNCT` near 99% with AnCora) and where it struggles (e.g. `SCONJ`, `PROPN`) due to lexical ambiguity and data sparsity.
- **Evaluate with at least five sentences:** `print_evaluation()` prints tagging samples for the first five test sentences, showing the original token sequence and the predicted tag sequence side by side.
- **Read another text and output PoS tags as a list of tuples:** The `--text` argument lets the user provide a new Spanish sentence from the command line. `tag_text()` tokenizes it and prints the output as a list of `(word, tag)` tuples.
- **Document use of AI:** The [Used AI](#used-ai) section explains how ChatGPT was used for assistance and lists the prompts used.

Main code files:

- [`tagger.py`](tagger.py): tokenization, corpus loading, train/test split, HMM training, evaluation, and command-line interface.
- [`spanish_corpus.py`](spanish_corpus.py): manual Spanish sentences, English translations, and tagged sentences.
- [`es_ancora-ud-train.conllu`](es_ancora-ud-train.conllu): optional large Universal Dependencies Spanish AnCora training corpus.
- [`requirements.txt`](requirements.txt): Python dependencies.

## Limitations and future work

The manual corpus is small, so accuracy is limited and many unseen words are difficult for the HMM. 

The tokenizer handles common punctuation but does not cover all Spanish tokenization cases, such as abbreviations, URLs, clitics, quotations, dates, and complex multiword expressions.

Unknown words at inference time are handled by NLTK's built-in Viterbi fallback; a suffix and capitalization-based feature scheme was explored but did not improve results on the AnCora test set due to high vocabulary coverage in training.
