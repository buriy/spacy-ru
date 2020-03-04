import os

import gensim
from spacy.pipeline.pipes import TextCategorizer, build_simple_cnn_text_classifier, build_bow_text_classifier, \
    build_text_classifier

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


def get_ft_vec():
    global VECTORS
    if VECTORS is None:
        fdir = os.path.dirname(os.path.dirname(__file__)) + '/data/vec/'
        VECTORS = gensim.models.KeyedVectors.load(fdir + 'vectors.bin')
    return VECTORS


def get_t2v(token_vector_width, embed_size, **cfg):
    vectors = get_ft_vec()
    t2v = my_tok_to_vec(token_vector_width, embed_size, vectors)
    return t2v


if __name__ == '__main__':
    import spacy

    nlp = spacy.blank('ru')
    doc = nlp('Привет вам, мужики!')
    doc2 = nlp('Зелёный день шагает по планете...')
    docs = [doc, doc2]
    t2v = get_t2v(96, 4000)
    r = t2v(docs)
    print(r)

    textcat = MyTextCategorizer(nlp.vocab, **{"exclusive_classes": True, "architecture": "simple_cnn"})
    nlp.add_pipe(textcat, name='textcat')
    for c in ['69-я параллель', 'Бизнес', 'Бывший СССР', 'Дом', 'Из жизни', 'Интернет и СМИ', 'Крым', 'Культпросвет ',
              'Культура', 'Мир', 'Наука и техника', 'Путешествия', 'Россия', 'Силовые структуры', 'Спорт', 'Ценности',
              'Экономика']:
        textcat.add_label(c)

    CFG = {'device': 0, 'cpu_count': 4}
    nlp.begin_training(**CFG)
    df = list(zip(docs, [1, 2]))
    # docs_iter = tqdm((nlp.tokenizer(x[0]) for x in df), total=len(df))
    # r = list(nlp.pipe(docs_iter))
    # print(r)
    r = textcat(doc)
