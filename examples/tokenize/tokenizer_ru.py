from spacy.lang.ru import Russian

from .sentences import sample_sentences
from .utils import print_tokens


if __name__ == '__main__':
    nlp = Russian()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print_tokens(doc)
