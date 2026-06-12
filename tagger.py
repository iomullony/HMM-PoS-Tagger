import argparse
import os
import re
from conllu import parse
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from spanish_corpus import (
    spanish_tagged_sentences,
    spanish_corpus_sentences,
    spanish_corpus_translations,
)

TRAIN_RATIO = 0.8
ANCORA_PATH = "es_ancora-ud-train.conllu"


def tokenize_sentence(sentence: str):
    sentence = sentence.replace("¿", " ¿ ").replace("¡", " ¡ ")
    sentence = re.sub(r"([.,;:!?()])", r" \1 ", sentence)
    tokens = [token for token in sentence.split() if token]
    return tokens


def split_corpus(sentences, ratio=TRAIN_RATIO):
    split_index = int(len(sentences) * ratio)
    return sentences[:split_index], sentences[split_index:]


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
            upos = token.get("upos") or "X"
            if form is None:
                continue
            tagged_tokens.append((form, upos))
        if tagged_tokens:
            tagged_sentences.append(tagged_tokens)

    return tagged_sentences


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

    return spanish_tagged_sentences


def train_hmm_tagger(train_sentences):
    trainer = hmm.HiddenMarkovModelTrainer()
    return trainer.train_supervised(
        train_sentences,
        estimator=lambda fdist, bins: LidstoneProbDist(fdist, 0.1, bins),
    )


def print_evaluation(tagger, train_sentences, test_sentences):
    accuracy = tagger.accuracy(test_sentences)
    print("Evaluation results")
    print("-------------------")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Accuracy:       {accuracy:.2%}\n")

    print("Test sentence tagging samples:")
    for sentence in test_sentences[:5]:
        tokens = [word for word, _ in sentence]
        tagged = tagger.tag(tokens)
        print("- original:", " ".join(tokens))
        print("  tagged:  ", tagged)
    print()


def print_corpus_translation_examples():
    print("Corpus examples with English translations:")
    for i in range(5):
        print(f"- {spanish_corpus_sentences[i]}")
        print(f"  {spanish_corpus_translations[i]}")
    print()


def tag_text(tagger, text):
    tokens = tokenize_sentence(text)
    print("Input sentence:", text)
    print("Tokens:", tokens)
    print("Tagged output:", tagger.tag(tokens))
    return tokens


if __name__ == "__main__":
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