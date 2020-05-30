from __future__ import unicode_literals, print_function

import sys
from pathlib import Path

import spacy
from spacy.lang.ru import Russian
from spacy.pipeline import Tagger, DependencyParser
from spacy.util import fix_random_seed, set_lang_class

from models.dep import MyDEP
from models.loadvec import get_ft_vec
from models.pos import MyPOS
from models.t2v import build_tok2vec
from training.corpora.syntagrus import get_syntagrus_example, get_syntagrus
from training.trainer import Trainer, Extractor
from utils.corpus import tag_morphology

CFG = {"device": 0, 'verbose': 1}
GPU_1 = "-g1" in sys.argv[1:]
if GPU_1:
    CFG["device"] = 1

TESTS = False
spacy.require_gpu(CFG['device'])

TEST_MODE = "--test" in sys.argv[1:]
if TEST_MODE:
    SynTagRus = get_syntagrus_example(Path("data/syntagrus/"))
else:
    SynTagRus = get_syntagrus(Path("data/syntagrus/"))


def create_pos(nlp, cls=MyPOS, labels=[], **opts):
    pos = cls(nlp.vocab, **opts)
    for e in labels:
        pos.add_label(e, tag_morphology(e))
    return pos


def create_dep(nlp, cls=MyDEP, labels=[], **opts):
    dep = cls(nlp.vocab, **opts)
    # for e in labels:
    #    dep.add_label(e)
    return dep


ft_vectors = get_ft_vec()
tok2vec = build_tok2vec(embed_size=2000, vectors={"word_vectors": ft_vectors})


def smoke_test():
    nlp = spacy.blank("ru")
    nlp.add_pipe(create_pos(nlp))
    nlp.add_pipe(create_dep(nlp))
    nlp.vocab.morphology.tag_map.clear()
    nlp.begin_training(tok2vec=tok2vec, **CFG)
    if TEST_MODE:
        print(nlp.pipeline)
    dep = nlp.get_pipe('parser')
    if TEST_MODE:
        print(dep(nlp.tokenizer("приветы всем")))


class Russian2(Russian):
    lang = "ru"


def train_spacy(nlp, epochs):
    # set_lang_class('ru2', Russian2)
    extractor = Extractor()
    cfg = {'tok2vec': tok2vec, **CFG}
    fix_random_seed()
    trainer = Trainer(nlp, SynTagRus.ds_train, SynTagRus.ds_test, extractor, **cfg)
    nlp.vocab.morphology.tag_map.clear()
    trainer.train(epochs=epochs)


def main():
    smoke_test()
    nlp = spacy.blank("ru")
    nlp.vocab.morphology.tag_map.clear()
    nlp.add_pipe(create_pos(nlp, labels=[]))
    nlp.add_pipe(create_dep(nlp, labels=[], config={'learn_tokens': False}))
    # nlp.add_pipe(create_pos(nlp, cls=Tagger, labels=SynTagRus.pos))
    # nlp.add_pipe(create_dep(nlp, cls=DependencyParser, labels=SynTagRus.dep, config={'learn_tokens': False}))
    if TEST_MODE:
        print(nlp.pipeline)
    # nlp.add_pipe(create_pos(nlp, labels=SynTagRus.pos))
    # nlp.add_pipe(create_dep(nlp, labels=SynTagRus.dep, config={'learn_tokens': False}))
    if TEST_MODE:
        train_spacy(nlp, epochs=5)
    else:
        train_spacy(nlp, epochs=50)


if __name__ == "__main__":
    main()
