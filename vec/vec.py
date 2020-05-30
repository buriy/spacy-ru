import os

import gensim
from spacy import util
from spacy._ml import flatten, PrecomputableAffine
from spacy.pipeline.pipes import (
    TextCategorizer,
    build_simple_cnn_text_classifier,
    build_bow_text_classifier,
    build_text_classifier,
    EntityRecognizer,
)
from spacy.syntax._parser_model import ParserModel
from thinc.api import chain
from thinc.neural import Model
from thinc.v2v import Affine

from .vec_utils import my_tok_to_vec

VECTORS = None


class MyTextCategorizer(TextCategorizer):
    @classmethod
    def Model(cls, nr_class=1, **cfg):
        print("Config:", cfg)
        embed_size = cfg.get("embed_size", 2000)
        token_vector_width = cfg.get("token_vector_width", 96)
        if cfg.get("architecture") == "simple_cnn":
            tok2vec = get_t2v(token_vector_width, embed_size, **cfg)
            return build_simple_cnn_text_classifier(tok2vec, nr_class, **cfg)
        elif cfg.get("architecture") == "bow":
            return build_bow_text_classifier(nr_class, **cfg)
        else:
            return build_text_classifier(nr_class, **cfg)


class MyNER(EntityRecognizer):
    @classmethod
    def Model(cls, nr_class, **cfg):
        depth = util.env_opt("parser_hidden_depth", cfg.get("hidden_depth", 1))
        subword_features = util.env_opt(
            "subword_features", cfg.get("subword_features", True)
        )
        conv_depth = util.env_opt("conv_depth", cfg.get("conv_depth", 4))
        conv_window = util.env_opt("conv_window", cfg.get("conv_depth", 1))
        t2v_pieces = util.env_opt("cnn_maxout_pieces", cfg.get("cnn_maxout_pieces", 3))
        bilstm_depth = util.env_opt("bilstm_depth", cfg.get("bilstm_depth", 0))
        self_attn_depth = util.env_opt("self_attn_depth", cfg.get("self_attn_depth", 0))
        assert depth == 1
        parser_maxout_pieces = util.env_opt(
            "parser_maxout_pieces", cfg.get("maxout_pieces", 2)
        )
        token_vector_width = util.env_opt(
            "token_vector_width", cfg.get("token_vector_width", 96)
        )
        hidden_width = util.env_opt("hidden_width", cfg.get("hidden_width", 64))
        embed_size = util.env_opt("embed_size", cfg.get("embed_size", 2000))
        tok2vec = get_t2v(
            token_vector_width,
            embed_size,
            conv_depth=conv_depth,
            conv_window=conv_window,
            cnn_maxout_pieces=t2v_pieces,
            subword_features=subword_features,
            bilstm_depth=bilstm_depth,
        )
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
            "hidden_depth": depth,
            "token_vector_width": token_vector_width,
            "hidden_width": hidden_width,
            "maxout_pieces": parser_maxout_pieces,
            "pretrained_vectors": None,
            "bilstm_depth": bilstm_depth,
            "self_attn_depth": self_attn_depth,
            "conv_depth": conv_depth,
            "conv_window": conv_window,
            "embed_size": embed_size,
            "cnn_maxout_pieces": t2v_pieces,
        }
        return ParserModel(tok2vec, lower, upper), cfg


def get_ft_vec():
    global VECTORS
    if VECTORS is None:
        fdir = os.path.dirname(os.path.dirname(__file__)) + "/data/vec/"
        VECTORS = gensim.models.KeyedVectors.load(fdir + "vectors.bin")
    return VECTORS


def get_t2v(token_vector_width, embed_size, **cfg):
    vectors = get_ft_vec()
    t2v = my_tok_to_vec(token_vector_width, embed_size, vectors)
    return t2v


if __name__ == "__main__":
    import spacy

    nlp = spacy.blank("ru")
    doc = nlp("Привет вам, мужики!")
    doc2 = nlp("Зелёный день шагает по планете...")
    docs = [doc, doc2]
    t2v = get_t2v(96, 4000)
    r = t2v(docs)
    print(r)

    textcat = MyTextCategorizer(
        nlp.vocab, **{"exclusive_classes": True, "architecture": "simple_cnn"}
    )
    nlp.add_pipe(textcat, name="textcat")
    for c in [
        "69-я параллель",
        "Бизнес",
        "Бывший СССР",
        "Дом",
        "Из жизни",
        "Интернет и СМИ",
        "Крым",
        "Культпросвет ",
        "Культура",
        "Мир",
        "Наука и техника",
        "Путешествия",
        "Россия",
        "Силовые структуры",
        "Спорт",
        "Ценности",
        "Экономика",
    ]:
        textcat.add_label(c)

    CFG = {"device": 0, "cpu_count": 4}
    nlp.begin_training(**CFG)
    df = list(zip(docs, [1, 2]))
    # docs_iter = tqdm((nlp.tokenizer(x[0]) for x in df), total=len(df))
    # r = list(nlp.pipe(docs_iter))
    # print(r)
    r = textcat(doc)
