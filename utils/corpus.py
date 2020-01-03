import random
import spacy
import spacy.gold
from itertools import chain
from tqdm.auto import tqdm


class Dataset:
    """
    Wraps
    Functionality:
     - Wraps dataset
     - Keeps all dataset labels, they are set up on dataset initialization
     - Keeps __len__
     - Allows dataset iteration
    """

    pos = None
    dep = None
    ner = None

    def __init__(self, lang=None):
        self.lang = lang
        self.pos = set()
        self.dep = set()
        self.ner = set()

    def __len__(self):
        return 0

    def iter(self, nlp, limit=None):
        raise NotImplementedError()

    def __iter__(self):
        nlp = spacy.blank(self.lang)
        return self.iter(nlp)

    def add_labels(self, gold):
        for tag in gold.tags:
            self.pos.add(tag)
        for tag in gold.labels:
            self.dep.add(tag)
        for tag in gold.ner:
            if tag != "-":
                self.ner.add(tag)


class RawDataset(Dataset):
    """
    RawDataset is the list of entries, each following this format:
    {
        'raw': "Raw text of the document.",
        'entities': NER entities following TRAIN_DATA format from the spacy docs: [(start_char, end_char, label), ...]
    } 
    """

    ds = None
    is_train = None

    def __init__(self, lang, ds, is_train=None):
        super(RawDataset, self).__init__(lang=lang)
        self.ds = ds
        self.is_train = is_train
        for doc, gold in self:
            self.add_labels(gold)

    def set_train_test(self, ds_train, ds_test):
        self.ds_train = ds_train
        self.ds_test = ds_test
        for p in chain(tqdm(ds_train), tqdm(ds_test)):
            gold = spacy.gold.GoldParse()
            gold.ner = p.get("entities", [])
            self.add_labels(gold)

    def iter(self, nlp, limit=None):
        ds_copy = self.ds[:]
        if self.is_train:
            random.shuffle(ds_copy)
        if limit:
            ds_copy = ds_copy[:limit]
        for dict_ in self.ds:
            doc = nlp(dict_["raw"])
            gold = spacy.gold.GoldParse()
            gold.ner = dict_.get("entities", [])
            yield doc, gold

    def __len__(self):
        return len(self.ds)


class GoldDataset(Dataset):
    ds = None
    is_train = None
    _len = None

    def __init__(self, lang, ds_gold, is_train=False):
        super(GoldDataset, self).__init__(lang=lang)
        self.ds = ds_gold
        self.is_train = is_train
        self._len = 0
        for d, gold in self:
            self.add_labels(gold)
            self._len += 1

    def iter(self, nlp, limit=None):
        if limit is not None:
            old_limit, self.ds.limit = self.ds.limit, limit
        try:
            ds = self.ds.train_docs(nlp) if self.is_train else self.ds.dev_docs(nlp)
            for doc, gold in ds:
                yield doc, gold
        finally:
            if limit is not None:
                self.ds.limit = old_limit

    def __len__(self):
        return self._len


class Corpus:
    """   
    A simple container for train and test datasets
    Supports .from_gold and .from_raw dataset building methods
    """

    ds_train = Dataset()
    ds_test = Dataset()

    def __init__(self, ds_train=None, ds_test=None):
        self.ds_train = ds_train
        self.ds_test = ds_test

    @staticmethod
    def from_gold(lang, ds_gold):
        train = GoldDataset(lang, ds_gold, is_train=True)
        test = GoldDataset(lang, ds_gold, is_train=False)
        return Corpus(train, test)

    @staticmethod
    def from_raw(lang, ds_train, ds_test):
        train = RawDataset(lang, ds_train)
        test = RawDataset(lang, ds_test)
        return Corpus(train, test)


def iter_corpora(corpora):
    for corpus in corpora:
        ds_list = (
            [corpus.ds_train, corpus.ds_test]
            if isinstance(corpus, Corpus)
            else [corpus]
        )
        for ds in ds_list:
            yield ds


def tag_morphology(tag):
    """
    >>> tag_morphology("NOUN__Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing")
    {'POS': 'NOUN', 'Animacy': 'Inan', 'Case': 'Nom', 'Gender': 'Neut', 'Number': 'Sing'}
    >>> tag_morphology("SYM___")
    {'POS': 'SYM'}
    """
    pos, parts = tag.split("__", 1)
    info = {"POS": pos}
    for p in parts.split("|"):
        if not p.strip("_"):
            continue
        if p.count("=") != 1:
            print("Weird case:", tag, ",", p)
        k, v = p.split("=", 1)
        info[k] = v
    return info
