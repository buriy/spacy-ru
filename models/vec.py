import numpy
from thinc.api import layerize
from thinc.v2v import Model


class FastTextVectors(Model):
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

        s = self.storage[-1]
        vectors = numpy.zeros((len(ids), self.nV), dtype=numpy.float32)
        for i, orth in enumerate(ids):
            w = s[int(orth)]
            vectors[i] = self.vecs[w]
        vectors = self.ops.xp.asarray(vectors)
        assert vectors.shape[0] == ids.shape[0]
        return vectors, None


def docs_to_vectors(embeddings):
    @layerize
    def DocVectors(docs, drop=0.0):
        batch = []
        for doc in docs:
            vector_list = []
            for token in doc:
                vector_list.append(embeddings[token.text])
            doc_vectors = numpy.stack(vector_list)
            batch.append(doc_vectors)
        return batch, None

    return DocVectors
