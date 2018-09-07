import string
from spacy.symbols import PRON
from .utils import print_tokens

class Linguistic(object):
    def __init__(self, nlp, stopwords, punctuations=string.punctuation):
        '''

        Args:
         nlp(spacy.language.Language): language parser
         stopwords(nltk.corpus.util.LazyCorpusLoader): stopwords for language
         punctuations(str): punctuations symbols
        '''
        self.punctuations = punctuations
        self.nlp = nlp
        self.stopwords = stopwords

    def cleanup(self, docs, debug_print_tokens=False, show_progress=None):
        '''
        Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
        Args:
            docs(list): list documents as str
            debug_print_tokens(bool): print tokens debug information
            show_progress: Ddecorate an iterable object
        Returns:
            clean_text(pd.Series): series clean documents.
        '''
        texts = []
        if show_progress:
            docs = show_progress(docs)
        for doc in docs:
            doc = self.nlp(doc)
            if debug_print_tokens:
                print_tokens(self.nlp, doc)
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != PRON]  # normalize words, clear pronouns form
            tokens = [tok for tok in tokens
                      if tok not in self.stopwords and tok not in self.punctuations]  # filtering tokens, clear all punctuations symbols and stop words
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return texts


if __name__ == '__main__':
    from ru2 import Russian2
    from nltk.corpus import stopwords
    import re
    import os
    import pandas as pd
    from tqdm import tqdm
    try:
        from .sentences import sample_sentences
    except:
        from sentences import sample_sentences

    ru_nlp = Russian2()
    ru_nlp.add_pipe(ru_nlp.create_pipe('sentencizer'))
    stopwords = stopwords.words('russian')
    lin = Linguistic(nlp=ru_nlp, stopwords=stopwords)
    ru_clean = pd.Series(lin.cleanup(list(re.split("[.!?]+", sample_sentences)), True, tqdm))
    ru_clean.to_csv('result.csv', encoding='utf-8')
    print("Result save to %s" % (os.path._getfullpathname('result.csv')))
