from spacy import util
from spacy._ml import flatten, PrecomputableAffine, Tok2Vec
from spacy.errors import TempErrors
from spacy.pipeline import DependencyParser
from spacy.syntax._parser_model import ParserModel
from thinc.api import chain
from thinc.neural import Model
from thinc.v2v import Affine


class MyDEP(DependencyParser):
    tok2vec_model = None

    @classmethod
    def set_tok2vec(cls, tok2vec):
        cls.tok2vec_model = tok2vec

    @classmethod
    def Model(cls, nr_class, **cfg):
        tok2vec = cfg.get("tok2vec", cls.tok2vec_model)
        assert tok2vec
        parser_maxout_pieces = util.env_opt(
            "parser_maxout_pieces", cfg.get("maxout_pieces", 2)
        )
        token_vector_width = util.env_opt(
            "token_vector_width", cfg.get("token_vector_width", 96)
        )
        hidden_width = util.env_opt("hidden_width", cfg.get("hidden_width", 64))
        embed_size = util.env_opt("embed_size", cfg.get("embed_size", 2000))
        tok2vec = chain(tok2vec, flatten)
        tok2vec.nO = token_vector_width
        lower = PrecomputableAffine(
            hidden_width,
            nF=cls.nr_feature,
            nI=token_vector_width,
            nP=parser_maxout_pieces,
        )
        lower.nP = parser_maxout_pieces

        with Model.use_device("cpu"):
            upper = Affine(nr_class, hidden_width, drop_factor=0.0)
        upper.W *= 0

        cfg = {
            "nr_class": nr_class,
            "token_vector_width": token_vector_width,
            "hidden_width": hidden_width,
            "maxout_pieces": parser_maxout_pieces,
            "pretrained_vectors": None,
            "embed_size": embed_size,
        }
        return ParserModel(tok2vec, lower, upper), cfg
