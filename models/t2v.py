from functools import reduce

from spacy import util
from spacy.attrs import ORTH, NORM, PREFIX, SUFFIX, SHAPE, LOWER
from thinc.api import chain, clone, concatenate, with_flatten
from thinc.api import uniqued
from thinc.i2v import HashEmbed
from thinc.misc import FeatureExtracter
from thinc.misc import LayerNorm as LN
from thinc.misc import Residual
from thinc.t2t import ExtractWindow
from thinc.v2v import Model, Maxout

from models.vec import FastTextVectors


class SaveDoc(Model):
    def __init__(self, storage):
        Model.__init__(self)
        self.storage = storage

    def begin_update(self, docs, drop=0.0):
        for doc in docs:
            if doc.vocab.strings not in self.storage:
                self.storage.append(doc.vocab.strings)
        return docs, _save_doc_bwd


def _save_doc_bwd(d_features, sgd=None):
    return d_features


def build_tok2vec(width=96, embed_size=2000, vectors={}, **kwargs):
    # Circular imports :(
    from spacy._ml import PyTorchBiLSTM

    assert vectors
    word_vectors = vectors.get("word_vectors", None)
    lemma_vectors = vectors.get("lemma_vectors", None)

    cnn_maxout_pieces = kwargs.get("cnn_maxout_pieces", 3)
    conv_depth = kwargs.get("conv_depth", 4)
    bilstm_depth = kwargs.get("bilstm_depth", 0)
    cols = [LOWER, NORM, PREFIX, SUFFIX, SHAPE, ORTH]
    storage = []
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        features = []
        feat_width = 0

        if word_vectors and True:
            e = FastTextVectors(storage, word_vectors, column=cols.index(LOWER))
            features.append(e)
            feat_width += e.nV

        if lemma_vectors and True:
            e = FastTextVectors(storage, lemma_vectors, column=cols.index(NORM))
            features.append(e)
            feat_width += e.nV

        norm = HashEmbed(
            width, embed_size, column=cols.index(NORM), name="embed_norm"
        )

        if True:  # kwargs.get("use_subwords", True):
            shape = HashEmbed(
                width, embed_size // 2, column=cols.index(SHAPE), name="embed_shape"
            )
            features.append(shape)
            feat_width += width

            features.append(norm)
            feat_width += width

            prefix = HashEmbed(
                width, embed_size // 2, column=cols.index(PREFIX), name="embed_prefix"
            )

            features.append(prefix)
            feat_width += width

            suffix = HashEmbed(
                width, embed_size // 2, column=cols.index(SUFFIX), name="embed_suffix"
            )

            features.append(suffix)
            feat_width += width

        embed = reduce(concatenate, features)
        embedding_layer = uniqued(
            embed >> LN(Maxout(width, feat_width, pieces=5)), column=cols.index(ORTH)
        )

        convolution_layer = Residual(
            ExtractWindow(nW=1)
            >> LN(Maxout(width, width * 3, pieces=cnn_maxout_pieces))
        )

        tok2vec = (
                SaveDoc(storage)
                >> FeatureExtracter(cols)
                >> with_flatten(
            embedding_layer >> convolution_layer ** conv_depth, pad=conv_depth
        )
        )

        if bilstm_depth >= 1:
            tok2vec = tok2vec >> PyTorchBiLSTM(width, width, bilstm_depth)
        # Work around thinc API limitations :(. TODO: Revise in Thinc 7

        tok2vec.nO = width
        tok2vec.embed = embedding_layer
    return tok2vec


def build_tok2vec_old(**cfg):
    subword_features = util.env_opt(
        "subword_features", cfg.get("subword_features", True)
    )
    conv_depth = util.env_opt("conv_depth", cfg.get("conv_depth", 4))
    conv_window = util.env_opt("conv_window", cfg.get("conv_depth", 1))
    t2v_pieces = util.env_opt("cnn_maxout_pieces", cfg.get("cnn_maxout_pieces", 3))
    bilstm_depth = util.env_opt("bilstm_depth", cfg.get("bilstm_depth", 0))
    token_vector_width = util.env_opt(
        "token_vector_width", cfg.get("token_vector_width", 96)
    )
    embed_size = util.env_opt("embed_size", cfg.get("embed_size", 2000))
    tok2vec = build_tok2vec(
        token_vector_width,
        embed_size,
        conv_depth=conv_depth,
        conv_window=conv_window,
        cnn_maxout_pieces=t2v_pieces,
        subword_features=subword_features,
        bilstm_depth=bilstm_depth,
    )
    return tok2vec
