#!/bin/sh
echo -n "$(basename $2)\t"
.venv/bin/python -t -m spacy evaluate -g 0 $1 $2 | egrep --color=never "Time|POS|UAS|LAS" | tr '\n' '\t'
echo ""