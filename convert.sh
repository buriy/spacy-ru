#!/bin/bash
echo 'Converting' $2
sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g; s/Tag=//g; s/|SpaceAfter=No//g' $2 >>/tmp/$$.conllu
.venv/bin/python -u -m spacy convert -n $1 -m /tmp/$$.conllu >$3
echo "OK"
