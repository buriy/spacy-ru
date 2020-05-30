import datetime
import pprint

import numpy
from spacy.gold import GoldParse, minibatch
from spacy.language import _pipe
from spacy.pipeline.pipes import Language
from spacy.scorer import Scorer
from spacy.util import compounding, minibatch_by_words, env_opt, load_model_from_path
from tqdm.auto import tqdm

from models.loadvec import get_ft_vec
from models.t2v import build_tok2vec
from utils.corpus import Dataset
from utils.pluck import pluck, pluck_dict


def tqdm_batches(batches, total=None, leave=True, **info):
    infostr = ", ".join([f"{k}={v}" for k, v in info.items()])
    if infostr:
        infostr = ", " + infostr
    ll = 0
    batch_iter = tqdm(total=total, leave=leave)
    for batch in batches:
        bl = len(batch)
        if bl > ll:
            batch_iter.set_description(f"bsz={bl}" + infostr)
            ll = bl
        yield batch
        batch_iter.update(bl)
    batch_iter.close()


def get_other_pipes(nlp, x):
    return [pipe for pipe in nlp.pipe_names if pipe not in x]


class Extractor:
    def make_val_batch(self, nlp, batch):
        return batch

    def make_train_batch(self, batch):
        return zip(*batch)


class RawExtractor(Extractor):
    def __init__(self, ds_field, gold_field, raw_field="raw"):
        self.field_raw = raw_field
        self.field_gold = gold_field
        self.field_ds = ds_field

    def make_gold(self, doc, parse):
        v = parse[self.field_ds]
        return doc, GoldParse(doc, **{self.field_gold: v})

    def make_val_batch(self, nlp, batch):
        docs = pluck(batch, self.field_raw)
        docs = nlp.pipe(docs)
        for doc, parse in zip(docs, batch):
            yield self.make_gold(doc, parse)

    def make_train_batch(self, batch):
        texts = pluck(batch, self.field_raw)
        anns = pluck_dict(batch, self.field_ds)
        return texts, anns


class Trainer:
    def __init__(self, nlp: Language, ds_train: Dataset, ds_valid: Dataset, extractor: Extractor, **cfg):
        self.cfg = cfg
        self.nlp = nlp
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.train_size = 100000
        self.valid_size = 2000
        self.valid_batch_size = 128
        self.dropout0 = 0.2
        self.dropout1 = 0.2
        self.dropout = self.dropout0
        self._optimizer = None
        self.models = ["tagger", "parser", "ner"]
        self.extractor = extractor
        self.state = {}
        self.logs = []
        self.start_date = str(datetime.datetime.utcnow().replace(microsecond=0)).replace(' ', '-').replace(':', '-')

    def evaluate_batches(self, batches, nlp_loaded):
        nlp = nlp_loaded

        scorer = Scorer()
        for batch in batches:
            docs, golds = zip(*self.extractor.make_val_batch(nlp, batch))
            golds = list(golds)
            kwargs = {}
            kwargs.setdefault("batch_size", self.valid_batch_size)
            for name, pipe in nlp.pipeline:
                if not hasattr(pipe, "pipe"):
                    docs = _pipe(docs, pipe, kwargs)
                else:
                    docs = pipe.pipe(docs, **kwargs)
            for doc, gold in zip(docs, golds):
                if not isinstance(gold, GoldParse):
                    gold = GoldParse(doc, **gold)
                scorer.score(doc, gold, **kwargs)
        return scorer.scores

    def evaluate(self, nlp_loaded):
        opts = dict(gold_preproc=1,
                    ignore_misaligned=True)
        item_count = min(self.valid_size, len(self.ds_valid))
        items = self.ds_valid.iter(nlp_loaded, limit=item_count, **opts)
        scorer = nlp_loaded.evaluate(self.ds_valid.iter(nlp_loaded, limit=item_count), verbose=False)
        return scorer.scores
        # batches = minibatch(items, self.valid_batch_size)
        # batches_ = tqdm_batches(batches, total=item_count, leave=False, **self.state)
        # return self.evaluate_batches(batches_, nlp_loaded)

    def train_batches(self, batches):
        model = self.nlp
        optimizer = self._begin()
        losses = {}
        n_docs = 0
        n_words = 0
        for batch in batches:
            texts, anns = self.extractor.make_train_batch(batch)
            model.update(texts, anns, drop=self.dropout, losses=losses, sgd=optimizer)
            n_docs += len(batch)
        loss = {k: numpy.log(1e-10 + (v / n_docs)) for k, v in losses.items()}
        loss.update({k + '_sum': v for k, v in losses.items()})
        meta = {
            "docs": n_docs,
            "words": n_words,
            "loss": loss,
        }
        return meta

    def train_epoch(self, batch_sizes):
        item_count = min(self.train_size, len(self.ds_train))
        opts = dict(noise_level=0,
                    orth_variant_level=0,
                    gold_preproc=1,
                    max_length=0,
                    ignore_misaligned=True)
        iter = self.ds_train.iter(self.nlp, **opts)
        batches = minibatch_by_words(iter, size=batch_sizes)
        # items = self.ds_train.iter(self.nlp, limit=item_count)
        # random.shuffle(items)
        # items = items[:50000]
        # batches = minibatch(items, batch_sizes)
        batches_ = tqdm_batches(batches, total=item_count, leave=False, **self.state)
        meta = self.train_batches(batches_)
        return meta

    def train(self, epochs):
        nlp = self.nlp
        if self._optimizer is None:
            self._optimizer = nlp.begin_training(lambda: self.ds_train.ds.train_tuples, **self.cfg)
            nlp._optimizer = None
        optimizer = self._begin()
        # batch_size = compounding(8., 16., 1.001)
        batch_size = compounding(
            env_opt("batch_from", 100.0),
            env_opt("batch_to", 400.0),
            env_opt("batch_compound", 1.001),
        )
        # batch_size = compounding(8., 16., 1.001)
        with nlp.disable_pipes(*get_other_pipes(nlp, self.models)):
            for epoch in range(epochs):
                self.dropout = self.dropout0 + (self.dropout1 - self.dropout0) * (epochs - epoch) / epochs
                self.state = {"epoch": epoch + 1, "dropout": self.dropout}
                meta = self.train_epoch(batch_sizes=batch_size)
                with nlp.use_params(optimizer.averages):
                    fpath = f'data/models/ru2-{self.start_date}-{self.state["epoch"]:02d}'
                    self.nlp.to_disk(fpath)
                    #ft_vectors = get_ft_vec()
                    #tok2vec = build_tok2vec(embed_size=2000, vectors={"word_vectors": ft_vectors})
                    #nlp_loaded = load_model_from_path(fpath, tok2vec=tok2vec)
                    # nlp_loaded.vocab.morphology.tag_map = nlp.vocab.morphology.tag_map
                    meta["epoch"] = epoch + 1
                    scores = self.evaluate(nlp)
                    scores.pop("las_per_type", None)
                    scores.pop("textcats_per_cat", None)
                    meta["scores_ner"] = {x: scores.pop(x, None) for x in
                                          ['ents_p', 'ents_r', 'ents_f', 'ents_per_type']}
                    del meta['scores_ner']
                    meta["scores_tagger"] = {x: scores.pop(x, None) for x in ['token_acc', 'tags_acc']}
                    meta["scores_parser"] = {x: scores.pop(x, None) for x in ['uas', 'las']}
                    # meta["scores"] = scores
                    pprint.pprint(meta)
                    self.logs.append(meta)

    def _begin(self):
        if self._optimizer is None:
            # for name, p in self.nlp.pipeline:
            #     if p.model is True:
            #         self._optimizer = self.nlp.begin_training(**self.cfg)
            #         break
            self._optimizer = self.nlp.begin_training(**self.cfg)
        return self._optimizer
