from collections import OrderedDict

from spacy._ml import link_vectors_to_models
from spacy.morphology import Morphology
from spacy.attrs import POS
from spacy.parts_of_speech import X
from spacy.pipeline import Tagger
from spacy.pipeline.pipes import Softmax
from thinc.api import chain, with_flatten


class MyPOS(Tagger):
    tok2vec_model = None

    @classmethod
    def set_tok2vec(cls, tok2vec):
        cls.tok2vec_model = tok2vec

    @classmethod
    def Model(cls, nr_class, **cfg):
        # print("CFG:", cfg)
        tok2vec = cfg.get("tok2vec", cls.tok2vec_model)
        assert tok2vec
        softmax = with_flatten(Softmax(nr_class, tok2vec.nO))
        model = chain(tok2vec, softmax)
        model.nI = None
        model.tok2vec = tok2vec
        model.softmax = softmax
        return model

    def begin_training(self, get_gold_tuples=lambda: [], pipeline=None, sgd=None,
                       **kwargs):
        # lemma_tables = ["lemma_rules", "lemma_index", "lemma_exc", "lemma_lookup"]
        # if not any(table in self.vocab.lookups for table in lemma_tables):
        #     user_warning(Warnings.W022)
        orig_tag_map = dict(self.vocab.morphology.tag_map)
        new_tag_map = OrderedDict()
        for raw_text, annots_brackets in get_gold_tuples():
            for annots, brackets in annots_brackets:
                ids, words, tags, heads, deps, ents = annots
                for tag in tags:
                    if tag in orig_tag_map:
                        new_tag_map[tag] = orig_tag_map[tag]
                    else:
                        new_tag_map[tag] = {POS: X}
        vocab = self.vocab
        if new_tag_map:
            vocab.morphology = Morphology(vocab.strings, new_tag_map,
                                          vocab.morphology.lemmatizer,
                                          exc=vocab.morphology.exc)
        # print("KWargs:", kwargs)
        self.cfg["tok2vec"] = kwargs.get("tok2vec")
        if self.model is True:
            for hp in ["token_vector_width", "conv_depth"]:
                if hp in kwargs:
                    self.cfg[hp] = kwargs[hp]
            self.model = self.Model(self.vocab.morphology.n_tags, **self.cfg)
        del self.cfg["tok2vec"]
        link_vectors_to_models(self.vocab)
        if sgd is None:
            sgd = self.create_optimizer()
        return sgd
