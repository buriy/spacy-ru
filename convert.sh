#!/bin/bash
s=$2
echo 'Converting' $s
fn="${s##*/}"
.venv/bin/python -u -m training.fix_conllu $2 >/tmp/$fn.$$.conllu
.venv/bin/python -u -m spacy convert -n $1 -m /tmp/$fn.$$.conllu $3
echo "OK"
