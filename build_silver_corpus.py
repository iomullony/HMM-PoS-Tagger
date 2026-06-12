import argparse
from pathlib import Path
import nltk
from nltk.corpus import cess_esp
from nltk import UnigramTagger, BigramTagger, DefaultTagger
import re


def tokenize_sentence(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace('¿', ' ¿ ').replace('¡', ' ¡ ')
    sentence = re.sub(r"([.,;:!?()\"'])", r' \1 ', sentence)
    tokens = [t for t in sentence.split() if t]
    return tokens



def map_cess_to_universal(tag):
    if not tag:
        return "X"
    t = tag.lower()
    if t.startswith("np"):
        return "PROPN"
    if t.startswith("n"):
        return "NOUN"
    if t.startswith("v"):
        return "VERB"
    if t.startswith("aq") or t.startswith("a"):
        return "ADJ"
    if t.startswith("da") or t.startswith("dd") or t.startswith("di"):
        return "DET"
    if t.startswith("pp") or t.startswith("p"):
        return "PRON"
    if t.startswith("rg") or t.startswith("r"):
        return "ADV"
    if t.startswith("c"):
        return "CONJ"
    if t.startswith("s"):
        return "PUNCT"
    if t.startswith("z") or t.startswith("w"):
        return "NUM"
    return "X"


def train_spanish_backoff():
    # Some NLTK installations may not include the full universal_tagset data.
    # Train on the original cess_esp tagset and map tags later with a heuristic.
    train_sents = cess_esp.tagged_sents()
    t0 = DefaultTagger('NOUN')
    t1 = UnigramTagger(train_sents, backoff=t0)
    t2 = BigramTagger(train_sents, backoff=t1)
    return t2


def tag_corpus_lines(tagger, lines):
    tagged = []
    for line in lines:
        toks = tokenize_sentence(line)
        if not toks:
            continue
        try:
            tagged_sent = tagger.tag(toks)
            # Map cess_esp tags to Universal tagset heuristically
            mapped = []
            for w, t in tagged_sent:
                ut = map_cess_to_universal(t)
                mapped.append((w, ut))
            tagged.append(mapped)
        except Exception:
            continue
    return tagged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='../midterm2/data/spa_news_2011_1M-sentences.txt')
    parser.add_argument('--sample', type=int, default=2000)
    parser.add_argument('--out', default='spanish_silver.py')
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    assert corpus_path.exists(), f"Corpus file not found: {corpus_path}"

    print('Training backoff Spanish tagger on cess_esp...')
    tagger = train_spanish_backoff()
    print('Done.')

    print(f'Reading up to {args.sample} sentences from corpus...')
    lines = []
    with corpus_path.open('r', encoding='utf8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
            if len(lines) >= args.sample:
                break
    print('Read', len(lines), 'lines')

    print('Tagging sampled lines...')

    # Debug: show tagging of first 3 lines
    for i,line in enumerate(lines[:3]):
        toks = tokenize_sentence(line)
        print('DBG line', i, 'tokens:', toks)
        try:
            print('DBG raw tag:', tagger.tag(toks))
        except Exception as e:
            print('DBG tagger error', e)

    # Inline tagging to avoid function-scope issues seen in some runtimes
    tagged = []
    for line in lines:
        toks = tokenize_sentence(line)
        if not toks:
            continue
        try:
            raw = tagger.tag(toks)
            mapped = [(w, map_cess_to_universal(t)) for w, t in raw]
            tagged.append(mapped)
        except Exception as e:
            print('tagging error', e)
            continue
    print('Tagged sentences:', len(tagged))

    out_path = Path(args.out)
    print('Writing silver corpus to', out_path)
    with out_path.open('w', encoding='utf8') as f:
        f.write('spanish_silver_tagged = [\n')
        for sent in tagged:
            f.write('    [')
            f.write(', '.join([f'("{w}", "{t}")' for w,t in sent]))
            f.write('],\n')
        f.write(']\n')
    print('Saved silver corpus.')

    # Quick evaluation: combine with existing masters corpus if present
    try:
        from spanish_corpus import spanish_tagged_sentences as manual
        print('Loaded manual corpus with', len(manual), 'sentences')
    except Exception:
        manual = []
    combined = manual + tagged
    print('Combined corpus size:', len(combined))

    # Train HMM briefly and show accuracy with 80/20 split
    from nltk.tag import hmm
    from nltk.probability import LidstoneProbDist
    split = int(len(combined)*0.8)
    train = combined[:split]
    test = combined[split:]
    print('Training HMM on combined corpus (this may take a moment)...')
    trainer = hmm.HiddenMarkovModelTrainer()
    try:
        model = trainer.train_supervised(train, estimator=lambda fdist, bins: LidstoneProbDist(fdist, 0.1, bins))
        acc = model.accuracy(test)
        print(f'Combined HMM accuracy on test set: {acc:.2%}')
    except Exception as e:
        print('HMM training failed:', e)

