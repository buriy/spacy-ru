import pandas
import spacy
from tabulate import tabulate


def pbool(x):
    return '+' if x else '-'


def entity_at(t):
    # print(t.i, t.idx, dir(t))
    entity = [e for e in t.doc.ents if e.start == t.i]
    if entity:
        return "{}: {}".format(t.ent_type_, entity[0].text)
    return ''


def print_tokens(nlp, doc):
    for s in doc.sents:
        print('Sentence: "{}"'.format(s))
        df = pandas.DataFrame(columns=['Shape', 'Vocab', 'POS', 'Text', 'Lemma', 'Entity', 'Dep', 'Head'],
                              data=[(t.shape_, pbool(t.orth_ in nlp.vocab), t.pos_,
                                     t.text, t.lemma_, entity_at(t), t.dep_, t.head) for t in s])
        print(tabulate(df, showindex=False, headers=df.columns))


sample_sentences = "Привет России и миру! Как твои дела? Сегодня неплохая погода."
if __name__ == '__main__':
    nlp = spacy.load('ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print("Sample sentences: {}".format(sample_sentences))
    print("\nResults for ru2 model: ")
    print_tokens(nlp, doc)
    nlp = spacy.load('ru2', disable=['tagger', 'parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(sample_sentences)
    print("\n"+"~"*70)
    print('\nSwitched to lemmatizer and POS from pymorphy2')
    print("Results for empty model: ")
    print_tokens(nlp, doc)
