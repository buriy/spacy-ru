import os
import random
import urllib.request
from itertools import chain
from typing import List

import spacy
import spacy.gold
from corus import load_factru
from corus.sources.factru import DEVSET as FACTRU_DEVSET, TESTSET as FACTRU_TESTSET
from corus.sources.factru import FactruSpan
from nerus import const as nerus_const
from tqdm.auto import tqdm

from .tqdm import TqdmUpTo


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
        for dict_ in ds_copy:
            doc = nlp(dict_["raw"])
            gold = spacy.gold.GoldParse(doc)
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
        train = RawDataset(lang, ds_train, is_train=True)
        test = RawDataset(lang, ds_test, is_train=False)
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


class FactRu(Corpus):

    @staticmethod
    def _resolve_data_path(data_path, download_if_not_exist):
        data_path = data_path or os.path.join(nerus_const.SOURCES_DIR, nerus_const.FACTRU_DIR, 'master.zip')
        print("FactRu data path: ", data_path)
        if not os.path.exists(data_path):
            if download_if_not_exist:
                os.makedirs(os.path.dirname(data_path))

                print("Download FactRu corpus to ", data_path)
                with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                              desc="FactRu corpus downloading") as tqdm_:
                    try:
                        urllib.request.urlretrieve(nerus_const.FACTRU_URL, data_path, reporthook=tqdm_.update_to)
                        tqdm_.total = tqdm_.n
                    except Exception as e:
                        os.remove(data_path)
                        raise e
                # TODO unpack in script
                raise Exception("{} file is downloaded, please unzip master.zip and restart script".format(data_path))
            else:
                raise FileExistsError("Source data for FactRuEval corpus is not exist: {}".format(data_path))
        return data_path

    @staticmethod
    def _load_list_dict(factru_data) -> List[dict]:
        output_data = []
        for element in factru_data:
            dict_ = {"raw": element.text}
            entities = []
            for fact in element.facts:
                for slot in fact.slots:
                    if isinstance(slot.value, FactruSpan):
                        span = slot.value
                        entities.append((span.start, span.stop, span.type))

                    elif not isinstance(slot.value, str):
                        for obj in slot.value.objects:
                            for span in obj.spans:
                                entities.append((span.start, span.stop, span.type))

            dict_["entities"] = entities
            output_data.append(dict_)
        return output_data

    @staticmethod
    def load(lang: str, data_path: str = None, download_if_not_exist: bool = True) -> Corpus:
        data_path = FactRu._resolve_data_path(data_path, download_if_not_exist)

        factru_dev_data = load_factru(data_path, sets=[FACTRU_DEVSET])
        factru_test_data = load_factru(data_path, sets=[FACTRU_TESTSET])

        train = RawDataset(lang, FactRu._load_list_dict(factru_dev_data), is_train=True)
        test = RawDataset(lang, FactRu._load_list_dict(factru_test_data), is_train=False)
        return Corpus(train, test)
