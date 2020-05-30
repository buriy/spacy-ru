from __future__ import unicode_literals

import numpy
from spacy.attrs import ID, ORTH, NORM, PREFIX, SUFFIX, SHAPE
from thinc.api import chain, clone, concatenate, with_flatten
from thinc.api import uniqued
from thinc.i2v import HashEmbed
from thinc.misc import FeatureExtracter
from thinc.misc import LayerNorm as LN
from thinc.misc import Residual
from thinc.t2t import ExtractWindow
from thinc.v2v import Model, Maxout


class Vectors(Model):
    name = "fasttext-vectors"

    def __init__(self, storage, vecs, drop_factor=0.0, column=0):
        Model.__init__(self)
        self.storage = storage
        self.vecs = vecs
        self.nV = 300
        self.drop_factor = drop_factor
        self.column = column

    def begin_update(self, ids, drop=0.0):
        if ids.ndim >= 2:
            ids = self.ops.xp.ascontiguousarray(ids[:, self.column])

        s = self.storage[0]
        vectors = numpy.zeros((len(ids), self.nV), dtype=numpy.float32)
        for i, orth in enumerate(ids):
            w = s[int(orth)]
            vectors[i] = self.vecs[w]
        vectors = self.ops.xp.asarray(vectors)
        assert vectors.shape[0] == ids.shape[0]
        return vectors, None


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


def my_tok_to_vec(width, embed_size, pretrained_vectors, **kwargs):
    # Circular imports :(
    from spacy._ml import PyTorchBiLSTM

    cnn_maxout_pieces = kwargs.get("cnn_maxout_pieces", 3)
    conv_depth = kwargs.get("conv_depth", 4)
    bilstm_depth = kwargs.get("bilstm_depth", 0)
    cols = [ID, NORM, PREFIX, SUFFIX, SHAPE, ORTH]
    storage = []
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        # norm = HashEmbed(width, embed_size, column=cols.index(NORM), name="embed_norm")
        # prefix = HashEmbed(
        #     width, embed_size // 2, column=cols.index(PREFIX), name="embed_prefix"
        # )
        # suffix = HashEmbed(
        #     width, embed_size // 2, column=cols.index(SUFFIX), name="embed_suffix"
        # )
        shape = HashEmbed(
            width, embed_size // 2, column=cols.index(SHAPE), name="embed_shape"
        )
        glove = Vectors(storage, pretrained_vectors, width, column=cols.index(NORM))
        vec_width = glove.nV

        embed = uniqued(
            (glove | shape) >> LN(Maxout(width, width + vec_width, pieces=3)),
            column=cols.index(ORTH),
        )

        convolution = Residual(
            ExtractWindow(nW=1)
            >> LN(Maxout(width, width * 3, pieces=cnn_maxout_pieces))
        )

        tok2vec = (
            SaveDoc(storage)
            >> FeatureExtracter(cols)
            >> with_flatten(embed >> convolution ** conv_depth, pad=conv_depth)
        )

        if bilstm_depth >= 1:
            tok2vec = tok2vec >> PyTorchBiLSTM(width, width, bilstm_depth)
        # Work around thinc API limitations :(. TODO: Revise in Thinc 7

        tok2vec.nO = width
        tok2vec.embed = embed
    return tok2vec
