import os
import unicodedata

from conllu import parse_incr
from tqdm.auto import tqdm


def count_roots(tokenlist):
    return sum(1 for tok in tokenlist if tok['deprel'] == 'root')


def count_reachable_nodes(tree_root):
    children_stack = tree_root.children
    reachable_nodes = {tree_root.token['id']}
    while children_stack:
        child = children_stack.pop()
        reachable_nodes.add(child.token['id'])
        children_stack.extend(child.children)

    return len(reachable_nodes)


def fix_tok_labels(token):
    deprel = token['deprel']
    upos = token['upos']

    if deprel == 'det':
        token['upos'] = 'DET'

    # Nummod is for "number phrases" only.
    if deprel == 'nummod' and upos not in ['NUM', 'NOUN', 'SYM']:
        token['upos'] = 'NUM'

    # Advmod is for adverbs, perhaps particles but not for prepositional phrases or clauses.
    if deprel == 'advmod' and upos not in ['ADV', 'ADJ', 'CCONJ', 'DET', 'PART', 'SYM']:
        token['upos'] = 'ADV'

    # Known expletives are pronouns. Determiners and particles are probably acceptable, too.
    if deprel == 'expl' and upos not in ['PRON', 'DET', 'PART']:
        token['upos'] = 'PRON'

    # Auxiliary verb/particle must be AUX.
    if deprel == 'aux':
        token['upos'] = 'AUX'

    # Copula is an auxiliary verb/particle (AUX) or a pronoun (PRON|DET).
    if deprel == 'cop' and upos not in ['AUX', 'PRON', 'DET', 'SYM']:
        token['upos'] = 'PRON'

    # Case is normally an adposition, maybe particle.
    # However, there are also secondary adpositions and they may have the original POS tag
    if deprel == 'case' and upos in ['PROPN', 'ADJ', 'PRON', 'DET', 'NUM', 'AUX']:
        token['upos'] = 'ADP'

    # Mark is normally a conjunction or adposition, maybe particle but definitely not a pronoun.
    if deprel == 'mark' and upos in ['NOUN', 'PROPN', 'ADJ', 'PRON', 'DET', 'NUM', 'VERB', 'AUX', 'INTJ']:
        token['upos'] = 'SCONJ'

    # Cc is a conjunction, possibly an adverb or particle.
    if deprel == 'cc' and upos in ['NOUN', 'PROPN', 'ADJ', 'PRON', 'DET', 'NUM', 'VERB', 'AUX', 'INTJ']:
        token['upos'] = 'CCONJ'

    if deprel == 'punct':
        token['upos'] = 'PUNCT'

    if upos == 'PUNCT' and deprel not in ['punct', 'root']:
        token['deprel'] = 'punct'


def fix_sent_labels(sentence):
    if len(sentence) < 1:
        return None

    if count_roots(sentence) != 1:
        # print(f"[sent_id = {sentence.metadata['sent_id']}] Multiple roots")
        return None

    tree = sentence.to_tree()

    if count_reachable_nodes(tree) != len(sentence):
        # print(f"[sent_id = {sentence.metadata['sent_id']}] Unreachable nodes or cycles")
        return None

    # combine unicode symbols with accents such as 'Й' into one
    sentence.metadata['text'] = unicodedata.normalize('NFC', sentence.metadata['text'])

    stext = sentence.metadata['text'] + '\n'
    next_start = 0
    for token in sentence:
        token['form'] = unicodedata.normalize('NFC', token['form'])

        # fix syntax
        fix_tok_labels(token)

        # infer SpaceAfter
        start = stext.find(token['form'], next_start)
        if not stext[start + len(token['form'])].isspace():
            token['misc']['SpaceAfter'] = 'No'
        next_start = start + len(token['form'])

    return sentence


def main():
    os.makedirs("data/nerus/chunks/", exist_ok=True)
    input_file = open("data/nerus/nerus_lenta.conllu", "r", encoding="utf-8")

    sentences = 0
    chunk_id = 0
    output_file = open(f"data/nerus/chunks/{chunk_id:03d}.conllu", "w", encoding="utf-8")

    for sentence in tqdm(parse_incr(input_file)):
        sentence = fix_sent_labels(sentence)
        if sentence is None:
            continue

        sentences += 1
        output_file.writelines(sentence.serialize())

        if sentences >= 50000:
            chunk_id += 1
            output_file = open(f"data/nerus/chunks/{chunk_id:03d}.conllu", "w", encoding="utf-8")
            sentences = 0

    input_file.close()
    output_file.close()


if __name__ == "__main__":
    main()
