import gzip
import json

from tqdm.auto import tqdm


def load_entries(fn):  # '../data/datasets/nerus.jsonl.gz'
    entries = []
    with gzip.open(fn, "r") as f:
        for line in tqdm(f):
            entry = json.loads(line)
            entries.append(entry)
    return entries
    # del entries


class Corpus:
    ents = set()
    ds_train = []
    ds_test = []


def get_nerus(root):
    NERUS = Corpus()
    NERUS.ents = {"ORG", "PER", "LOC"}
    NERUS.ds_test = load_entries(root / "nerus_test.jsonl.gz")
    NERUS.ds_train = load_entries(root / "nerus_train.jsonl.gz")

    # BUGFIX for v0.5
    # NERUS.ds_test = [{'raw': x['raw'], 'entities': x['entries']} for x in tqdm(NERUS.ds_test)]
    # NERUS.ds_train = [{'raw': x['raw'], 'entities': x['entries']} for x in tqdm(NERUS.ds_train)]

    # BUGFIX for v0.9
    NERUS.ner = NERUS.ents
    return NERUS


def get_nerus_example(root):
    NERUS = Corpus()
    NERUS.ents = {"ORG", "PER", "LOC"}
    NERUS.ds_train = load_entries(root / "nerus_train_example.jsonl.gz")
    NERUS.ds_test = load_entries(root / "nerus_test_example.jsonl.gz")

    # BUGFIX for v0.5
    # NERUS.ds_test = [{'raw': x['raw'], 'entities': x['entries']} for x in tqdm(NERUS.ds_test)]
    # NERUS.ds_train = [{'raw': x['raw'], 'entities': x['entries']} for x in tqdm(NERUS.ds_train)]

    # BUGFIX for v0.9
    NERUS.ner = NERUS.ents
    return NERUS
