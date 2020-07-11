#!/bin/bash
echo 'Converting' $1
sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $1 >>$1~.conllu
.venv/bin/python -t -m spacy convert -n 10 -m $1~.conllu >$2
echo "OK"