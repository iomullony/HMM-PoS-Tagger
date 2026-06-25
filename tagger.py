import argparse
import os
import random
import re
from conllu import parse
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from collections import defaultdict

from spanish_corpus import (
    spanish_tagged_sentences,
    spanish_corpus_sentences,
    spanish_corpus_translations,
)

TRAIN_RATIO = 0.8   # 80% for training and 20% for testing
RANDOM_SEED = 42    # Randomized split 
ANCORA_PATH = "es_ancora-ud-train.conllu"


# Converts a raw sentence string into a list of tokens
def tokenize_sentence(sentence: str):
    sentence = sentence.replace("¿", " ¿ ").replace("¡", " ¡ ")
    sentence = re.sub(r"([.,;:!?()])", r" \1 ", sentence)

    tokens = [token for token in sentence.split() if token]

    return tokens


# Divides the tagged corpus into training and testing portions
def split_corpus(sentences, ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    shuffled_sentences = list(sentences)
    random.Random(seed).shuffle(shuffled_sentences)
    split_index = int(len(shuffled_sentences) * ratio)

    return shuffled_sentences[:split_index], shuffled_sentences[split_index:]


# Loads the optional larger corpus from AnCora
def load_ancora_tagged_sentences(conllu_path=ANCORA_PATH):
    if not os.path.exists(conllu_path):
        return []

    with open(conllu_path, encoding="utf8") as f:
        corpus = parse(f.read())

    tagged_sentences = []
    for sent in corpus:
        tagged_tokens = []
        for token in sent:
            form = token.get("form")
            upos = token.get("upos")
            if form is None or not upos or upos == "_":
                continue
            tagged_tokens.append((form, upos))
        if tagged_tokens:
            tagged_sentences.append(tagged_tokens)

    return tagged_sentences


# Decides which corpus to use
def get_training_corpus(use_ancora=False):
    if use_ancora:
        ancora_sentences = load_ancora_tagged_sentences()
        if ancora_sentences:
            print(f"Loaded {len(ancora_sentences)} UD AnCora sentences.")
            return ancora_sentences

        print(
            f"Warning: {ANCORA_PATH} not found or empty. "
            "Falling back to the small manual Spanish corpus."
        )

    # If the AnCora is not being used, use the manual corpus
    return spanish_tagged_sentences


def train_hmm_tagger(train_sentences):
    # NLTK HMM trainer
    trainer = hmm.HiddenMarkovModelTrainer() 

    # Trains the model using supervised learning
    return trainer.train_supervised(
        train_sentences,
        estimator=lambda fdist, bins: LidstoneProbDist(fdist, 0.1, bins),
    )


# This is a more detailed evaluation
def print_detailed_evaluation(tagger, test_sentences):
    correct = defaultdict(int)
    total   = defaultdict(int)

    for sentence in test_sentences:
        tokens   = [word for word, _ in sentence]
        gold     = [tag  for _, tag  in sentence]
        predicted = [tag  for _, tag  in tagger.tag(tokens)]

        for g, p in zip(gold, predicted):
            total[g] += 1
            if g == p:
                correct[g] += 1

    print("Per-tag accuracy:")
    print(f"  {'Tag':<12} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print("  " + "-" * 40)
    for tag in sorted(total):
        acc = correct[tag] / total[tag] if total[tag] else 0
        print(f"  {tag:<12} {correct[tag]:>7} {total[tag]:>7} {acc:>9.2%}")
    print()


# Evaluates the trained tagger on the test set
def print_evaluation(tagger, train_sentences, test_sentences):
    accuracy = tagger.accuracy(test_sentences)
    print("Evaluation results")
    print("-------------------")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Accuracy:       {accuracy:.2%}\n")
    print_detailed_evaluation(tagger, test_sentences)

    print("Test sentence tagging samples:")
    for sentence in test_sentences[:5]:
        tokens = [word for word, _ in sentence]
        tagged = tagger.tag(tokens)
        print("- original:", " ".join(tokens))
        print("  tagged:  ", tagged)
    print()


# Prints Spanish sentences and their English translations from spanish_corpus.py
def print_corpus_translation_examples():
    print("Corpus examples with English translations:")
    for i in range(5):
        print(f"- {spanish_corpus_sentences[i]}")
        print(f"  {spanish_corpus_translations[i]}")
    print()


# Tags a custom sentence
def tag_text(tagger, text):
    tokens = tokenize_sentence(text)

    print("Input sentence:", text)
    print("Tokens:", tokens)
    print("Tagged output:", tagger.tag(tokens))

    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Spanish HMM PoS-tagger."
    )
    parser.add_argument(
        "--use-ancora",
        action="store_true",
        help="Use local UD Spanish AnCora data from es_ancora-ud-train.conllu.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="A Spanish sentence to tag. If omitted, the script tags a default sample.",
    )
    parser.add_argument(
        "--show-translations",
        action="store_true",
        help="Show example corpus sentences and their English translations.",
    )
    args = parser.parse_args()

    tagged_sentences = get_training_corpus(use_ancora=args.use_ancora)
    train_sentences, test_sentences = split_corpus(tagged_sentences)
    tagger = train_hmm_tagger(train_sentences)

    print_evaluation(tagger, train_sentences, test_sentences)
    
    if args.show_translations:
        print_corpus_translation_examples()

    if args.text:
        print("Custom tagging:")
        tag_text(tagger, args.text)
    else:
        sample_text = (
            spanish_corpus_sentences[-1]
            if not args.use_ancora
            else "¿Cómo estás hoy?"
        )
        print("Default tagging example:")
        tag_text(tagger, sample_text)


if __name__ == "__main__":
    main()
