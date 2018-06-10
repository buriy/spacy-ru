def print_tokens(doc):
    for s in doc.sents:
        print('Sentence: "{}"'.format(s))
        for token in s:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(
                token.shape_, token.pos_, token.is_punct, token.text, token.lemma_, token.tag_ 
            ))
