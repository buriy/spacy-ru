import Stemmer
from spacy.tokenizer import Tokenizer

class StemmingTokenizer(Tokenizer):
    def __init__(self, vocab, **kw):
        assert vocab is not None
        super(StemmingTokenizer, self).__init__(vocab, **kw)
        self.stemmer = Stemmer.Stemmer(vocab.lang)

    def __call__(self, text):
        tokens = super(StemmingTokenizer, self).__call__(text)
        for t in tokens:
            t.norm_ = self.stemmer.stemWord(t.lower_)
        return tokens
