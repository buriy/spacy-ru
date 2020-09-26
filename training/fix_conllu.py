import sys
import typing
import unicodedata

from conllu import parse_incr
from tqdm.auto import tqdm


def fix_misc(misc):
    if misc and isinstance(misc, dict):
        if 'Tag' in misc:
            return misc['Tag']
        return None
    return misc


def fix_feats(feats):
    if not feats:
        return feats
    new_feats = {}
    for feat, val in feats.items():
        if feat == 'Variant':
            feat = 'StyleVariant'
        if val == '1':
            val = 'First'
        if val == '2':
            val = 'Second'
        if val == '3':
            val = 'Third'
        new_feats[feat] = val
    return new_feats


def main(filenames: typing.List[str]):
    if len(filenames) == 0:
        print("Usage: fix_conllu <input1.conllu> <input2.conllu> ...", file=sys.stderr)
        return

    for fn in filenames:
        with open(fn, 'rt') as input_file:
            for sentence in tqdm(parse_incr(input_file)):
                for token in sentence:
                    form_ = unicodedata.normalize('NFC', token['form'])
                    assert form_ == token['form'], 'This file needs to be fixed.'
                    token['xpos'] = None
                    token['misc'] = fix_misc(token.get('misc', None))
                    if 'misc' in token:
                        assert token['misc'] == token.get('misc', None)
                    token['feats'] = fix_feats(token['feats'])

                sys.stdout.writelines(sentence.serialize())


if __name__ == "__main__":
    main(sys.argv[1:])
