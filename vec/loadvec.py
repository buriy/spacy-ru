from __future__ import unicode_literals

import numpy
from spacy.attrs import ID, ORTH, NORM, PREFIX, SUFFIX, SHAPE
from thinc.api import chain, clone, concatenate, with_flatten, layerize
from thinc.api import uniqued
from thinc.i2v import HashEmbed
from thinc.misc import FeatureExtracter
from thinc.misc import LayerNorm as LN
from thinc.misc import Residual
from thinc.t2t import ExtractWindow
from thinc.v2v import Model, Maxout

VECTORS = None


def get_ft_vec():
    global VECTORS
    if VECTORS is None:
        VECTORS = gensim.models.KeyedVectors.load(
            "/dsdata/models/vec/web_snakers41_pruned_3.bin"
        )
    return VECTORS


@layerize
def DocVectors(docs, drop=0.0):
    ft_vec = get_ft_vec()
    batch = []
    for doc in docs:
        vector_list = []
        for token in doc:
            vector_list.append(ft_vec[token.text])
        vectors = numpy.stack(vector_list)
        batch.append(vectors)
    return batch, None


# import numpy
# class FastTextEmbeddingBag(EmbeddingBag):
#     def __init__(self, model_path):
#         self.model = load_model(model_path)
#         input_matrix = self.model.get_input_matrix()
#         input_matrix_shape = input_matrix.shape
#         super().__init__(input_matrix_shape[0], input_matrix_shape[1])
#         self.weight.data.copy_(torch.FloatTensor(input_matrix))
#
#     def forward(self, words):
#         word_subinds = numpy.empty([0], dtype=numpy.int64)
#         word_offsets = [0]
#         for word in words:
#             _, subinds = self.model.get_subwords(word)
#             word_subinds = numpy.concatenate((word_subinds, subinds))
#             word_offsets.append(word_offsets[-1] + len(subinds))
#         word_offsets = word_offsets[:-1]
#         ind = torch.LongTensor(word_subinds)
#         offsets = torch.LongTensor(word_offsets)
#         return super().forward(ind, offsets)
#
#
def Tok2Vec(width, embed_size, **kwargs):
    # Circular imports :(
    from spacy._ml import PyTorchBiLSTM

    cnn_maxout_pieces = kwargs.get("cnn_maxout_pieces", 3)
    conv_depth = kwargs.get("conv_depth", 4)
    bilstm_depth = kwargs.get("bilstm_depth", 0)
    cols = [ID, NORM, PREFIX, SUFFIX, SHAPE, ORTH]
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        norm = HashEmbed(width, embed_size, column=cols.index(NORM), name="embed_norm")
        prefix = HashEmbed(
            width, embed_size // 2, column=cols.index(PREFIX), name="embed_prefix"
        )
        suffix = HashEmbed(
            width, embed_size // 2, column=cols.index(SUFFIX), name="embed_suffix"
        )
        shape = HashEmbed(
            width, embed_size // 2, column=cols.index(SHAPE), name="embed_shape"
        )
        glove = SpacyVectors()
        vec_width = 300

        embed = uniqued(
            (glove | norm | prefix | suffix | shape)
            >> LN(Maxout(width, width * 4 + vec_width, pieces=3)),
            column=cols.index(ORTH),
        )

        convolution = Residual(
            ExtractWindow(nW=1)
            >> LN(Maxout(width, width * 3, pieces=cnn_maxout_pieces))
        )

        tok2vec = FeatureExtracter(cols) >> with_flatten(
            embed >> convolution ** conv_depth, pad=conv_depth
        )

        if bilstm_depth >= 1:
            tok2vec = tok2vec >> PyTorchBiLSTM(width, width, bilstm_depth)
        # Work around thinc API limitations :(. TODO: Revise in Thinc 7
        tok2vec.nO = width
        tok2vec.embed = embed
    return tok2vec


t2v = Tok2Vec(96, 300, pretrained_vectors=vectors)

if __name__ == "__main__":
    import gensim

    vectors = gensim.models.KeyedVectors.load("../../ru2/ru_300.fasttext/model.model")
    t2v = Tok2Vec(96, 300, pretrained_vectors=vectors)
