from spacy.gold import GoldCorpus

from utils.corpus import Corpus


def get_syntagrus(root):
    g = GoldCorpus(
        root / "ru_syntagrus-ud-train.json",
        root / "ru_syntagrus-ud-test.json",
    )
    g.limit = None
    SynTagRus = Corpus.from_gold("ru", g)
    SynTagRus.pos = set(SynTagRus.ds_train.pos)
    SynTagRus.dep = set(SynTagRus.ds_train.dep)
    return SynTagRus


def get_syntagrus_example(root):
    g = GoldCorpus(
        root / "ru_syntagrus-ud-test.json",
        root / "ru_syntagrus-ud-test.json",
    )
    g.limit = None
    SynTagRus = Corpus.from_gold("ru", g)
    SynTagRus.pos = set(SynTagRus.ds_train.pos)
    SynTagRus.dep = set(SynTagRus.ds_train.dep)
    return SynTagRus
