from .sentences import sample_sentences
from .utils import print_tokens
import spacy

if __name__ == '__main__':
    nlp = spacy.load('ru2')
#    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
#     nlp.add_pipe(nlp.create_pipe('tagger'))
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print_tokens(nlp, doc)
    #for e in doc.ents:
    #    print('"',e.text,'"', e.label_)
