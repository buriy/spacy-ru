from __future__ import unicode_literals, print_function

import sys
from pathlib import Path

import spacy

from models.loadvec import get_ft_vec
from models.ner import MyNER
from models.t2v import build_tok2vec
from training.ner.nerus import get_nerus, get_nerus_example
from training.ner.trainer import Trainer, Extractor

CFG = {"device": 0, "cpu_count": 4}
TESTS = False
spacy.require_gpu()

GPU_1 = "-g 1" in sys.argv[1:]
if GPU_1:
    CFG["device"] = 1
TEST_MODE = "--test" in sys.argv[1:]
if TEST_MODE:
    NERUS = get_nerus_example(Path("data/nerus/"))
else:
    NERUS = get_nerus(Path("data/nerus/"))


def create_ner(nlp):
    return MyNER(nlp.vocab)


ft_vectors = get_ft_vec()
tok2vec = build_tok2vec(embed_size=2000, vectors={"word_vectors": ft_vectors})


def smoke_test():
    nlp = spacy.blank("ru")
    ner = create_ner(nlp)
    nlp.add_pipe(ner)
    nlp.begin_training(pretrained_vectors={'tok2vec': tok2vec}, **CFG)
    print(ner(nlp.tokenizer("приветы всем")))


def train_ner(nlp, epochs):
    extractor = Extractor("entities", "entities")
    cfg = {"tok2vec": tok2vec, **CFG}
    trainer = Trainer(nlp, NERUS.ds_train, NERUS.ds_test, extractor, **cfg)
    trainer.train(epochs=epochs)


def main():
    smoke_test()
    nlp = spacy.blank("ru")
    ner = create_ner(nlp)
    for e in NERUS.ents:
        ner.add_label(e)
    nlp.add_pipe(ner)
    if TEST_MODE:
        train_ner(nlp, epochs=5)
    else:
        train_ner(nlp, epochs=50)


if __name__ == "__main__":
    main()
