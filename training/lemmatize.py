import pymorphy2

ma = pymorphy2.MorphAnalyzer()


def ltag(a):
    return str(a.tag).split(' ')[0]


parses = ma.parse('кошка');
for p in parses:
    print(ltag(p), 'score=', p.score, '::')
    for f in p.lexeme:
        if ltag(f) == ltag(p):
            print(f.tag, str(f.word))
    print('-' * 40)
