# encoding: utf8
from __future__ import unicode_literals, print_function
from spacy.lang.ru import RussianDefaults, Russian
from ru2.lemmatizer import RussianLemmatizer
from .syntax_iterators import SYNTAX_ITERATORS

class Russian2Defaults(RussianDefaults):
    syntax_iterators = SYNTAX_ITERATORS
    @classmethod
    def create_lemmatizer(cls, nlp=None):
        return RussianLemmatizer()


class Russian2(Russian):
    lang = 'ru'
    Defaults = Russian2Defaults

def load_ru2(path, exclude=[]):
    nlp = Russian2()
    nlp.from_disk(path)
    nlp.disable_pipes(*exclude)
    return nlp

__all__ = ['Russian2', 'load_ru2']
