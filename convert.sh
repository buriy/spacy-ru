#!/bin/bash
echo 'Converting' $2
sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $2 >>$2~.conllu
.venv/bin/python -u -m spacy convert -n $1 -m $2~.conllu >$3
echo "OK"