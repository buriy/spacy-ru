from pathlib import Path

import numpy as np
import plac
import pymorphy2
from navec import Navec
from spacy.language import Language
from spacy.vectors import Vectors
from spacy.vocab import Vocab


class RussianLanguage(Language):
    lang = 'ru'


@plac.annotations(
    model=("Navec model archive.", "option", "m", str),
    output_dir=("Model output directory", "option", "o", Path)
)
def main(model='../data/vec/navec_hudlit_v1_12B_500K_300d_100q.tar', output_dir='../data/navec_hudlit.test.model'):
    morph_analyzer = pymorphy2.MorphAnalyzer()
    navec_model = Navec.load(model)
    print('Loaded model')

    words = navec_model.vocab.words
    vocabulary = Vocab()
    vocabulary.vectors = Vectors(shape=(len(words), 300), name='navec_lex')

    added_vectors = 0
    added_lexemes = 0
    for word in words:
        # retrieve unique lexemes for a given word
        parsed_word = morph_analyzer.parse(word)
        lexemes = set([lexeme.word for lexeme in parsed_word[0].lexeme])

        # check if at least 1 lexeme is absent in a vocabulary (returned list will have -1 values)
        rows = vocabulary.vectors.find(keys=lexemes)
        if any(row == -1 for row in rows):
            # find a vector for each lexeme if exists
            vectors = []
            for lexeme in lexemes:
                vector = navec_model.get(lexeme)
                if vector is not None:
                    vectors.append(vector)

            if len(vectors) > 0:
                # make sure we don't have duplicates before computing mean vector
                mean_vector = np.unique(vectors, axis=0).mean(axis=0)
                # filter lexemes by indices which reflect -1 values returned by vocab's rows lookup
                unique_lexemes = [lexeme for idx, lexeme in enumerate(lexemes) if rows[idx] == -1]
                # add a new vector and a first hash from paradigm
                vector_row = vocabulary.vectors.add(unique_lexemes.pop(), vector=mean_vector)
                # map other lexemes' hashes with the above vector
                for lexeme in unique_lexemes:
                    vocabulary.vectors.add(lexeme, row=vector_row)
                # collect stats
                added_lexemes += (len(unique_lexemes) + 1)
                added_vectors += 1

    print('Vectors:', added_vectors, 'Lexemes:', added_lexemes)
    removed_vectors = vocabulary.vectors.resize(shape=(added_vectors, 300))
    print('Resized vocabulary to', added_vectors)
    print('Removed vectors:', removed_vectors)

    nlp = RussianLanguage(vocabulary)
    nlp.to_disk(output_dir)
    print('Saved model to disk')


if __name__ == "__main__":
    plac.call(main)
