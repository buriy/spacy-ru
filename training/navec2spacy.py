from pathlib import Path

import numpy as np
import plac
import pymorphy2
from navec import Navec
from spacy.language import Language
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from tqdm.auto import tqdm


class RussianLanguage(Language):
    lang = 'ru'


def tags_from(word):
    return str(word.tag).split(' ')[0]


@plac.annotations(
    model=("Navec model archive.", "option", "m", str),
    output_dir=("Model output directory", "option", "o", Path)
)
def main(model='navec_hudlit_v1_12B_500K_300d_100q.tar',
         output_dir='./model'):
    morph_analyzer = pymorphy2.MorphAnalyzer()
    navec_model = Navec.load(model)
    print('Loaded model')

    known_tags = [tag for tag in morph_analyzer.TagClass.KNOWN_GRAMMEMES]
    vectors_dims = 300 + len(known_tags)
    words = navec_model.vocab.words

    vocabulary = Vocab()
    vocabulary.vectors = Vectors(shape=(len(words), vectors_dims), name='navec_lex')

    added_vectors = 0
    added_lexemes = 0
    for word in tqdm(words):
        parsed_word = morph_analyzer.parse(word)
        word_tags = tags_from(parsed_word[0])
        # retrieve unique lexemes for a given word filtered by similar tags
        lexemes = set([lexeme.word for lexeme in parsed_word[0].lexeme if word_tags == tags_from(lexeme)])

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
                tags_cipher = [+(tag in word_tags) for tag in known_tags]
                # create mean vector merged with tags
                mean_vector = np.append(np.unique(vectors, axis=0).mean(axis=0), tags_cipher, axis=0)
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
    removed_vectors = vocabulary.vectors.resize(shape=(added_vectors, vectors_dims))
    print('Resized vocabulary to', added_vectors)
    print('Removed vectors:', removed_vectors)

    nlp = RussianLanguage(vocabulary)
    nlp.to_disk(output_dir)
    print('Saved model to disk')


if __name__ == "__main__":
    plac.call(main)
