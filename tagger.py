import argparse
import re
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from spanish_corpus import spanish_tagged_sentences, spanish_corpus_sentences, spanish_corpus_translations

TRAIN_RATIO = 0.8


def tokenize_sentence(sentence: str):
    sentence = sentence.replace("¿", " ¿ ").replace("¡", " ¡ ")
    sentence = re.sub(r'([.,;:!?()])', r' \1 ', sentence)
    tokens = [token for token in sentence.split() if token]
    return tokens


def split_corpus(sentences, ratio=TRAIN_RATIO):
    split_index = int(len(sentences) * ratio)
    return sentences[:split_index], sentences[split_index:]


def train_hmm_tagger(train_sentences):
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(
        train_sentences,
        estimator=lambda fdist, bins: LidstoneProbDist(fdist, 0.1, bins),
    )
    return tagger


def print_evaluation(tagger, train_sentences, test_sentences):
    accuracy = tagger.accuracy(test_sentences)
    print(f"Evaluation results")
    print(f"-------------------")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Accuracy:       {accuracy:.2%}\n")

    print("Test sentence tagging samples:")
    for sentence in test_sentences[:5]:
        tokens = [word for word, _tag in sentence]
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
    tagged = tagger.tag(tokens)
    print("Tagged output:", tagged)
    return tagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Spanish HMM PoS-tagger using a small annotated corpus."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="A Spanish sentence to tag. If omitted, the script tags a default sample."
    )
    parser.add_argument(
        "--show-translations",
        action="store_true",
        help="Show example corpus sentences and their English translations."
    )
    args = parser.parse_args()

    train_sentences, test_sentences = split_corpus(spanish_tagged_sentences)
    tagger = train_hmm_tagger(train_sentences)

    print_evaluation(tagger, train_sentences, test_sentences)
    if args.show_translations:
        print_corpus_translation_examples()

    if args.text:
        print("Custom tagging:")
        tag_text(tagger, args.text)
    else:
        sample_text = spanish_corpus_sentences[-1]
        print("Default tagging example:")
        tag_text(tagger, sample_text)
