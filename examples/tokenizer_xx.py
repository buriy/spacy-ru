import spacy

from .sentences import sample_sentences
from .utils import print_tokens


if __name__ == '__main__':
    nlp = spacy.load('xx')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences.replace('. ', ' . '))
    print_tokens(nlp, doc)
    for e in doc.ents:
        print(e)
