import pandas
from tabulate import tabulate

def pbool(x):
    return '+' if x else '-'

def print_tokens(nlp, doc):
    for s in doc.sents:
        print('Sentence: "{}"'.format(s))
        df = pandas.DataFrame(columns=['Shape', 'Vocab', 'POS', 'Text', 'Tag', 'Lemma', 'Dep', 'Head'], 
                             data=[(t.shape_, pbool(t.orth_ in nlp.vocab), t.pos_, 
                                    t.text, t.tag_, t.lemma_, t.dep_, t.head) for t in s])
        print(tabulate(df, showindex=False, headers=df.columns))
