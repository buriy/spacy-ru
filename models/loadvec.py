import os

import gensim

VECTORS = None


def get_ft_vec():
    global VECTORS
    if VECTORS is None:
        fdir = os.path.dirname(os.path.dirname(__file__)) + "/data/vec/"
        VECTORS = gensim.models.KeyedVectors.load(fdir + "vectors.bin")
    return VECTORS
