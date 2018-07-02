from .sentences import sample_sentences
from .utils import print_tokens
from ru2 import Russian2

if __name__ == '__main__':
    nlp = Russian2()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
#     nlp.add_pipe(nlp.create_pipe('tagger'))
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print_tokens(doc)
