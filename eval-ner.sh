#!/bin/sh
SRC=$1
DEST=$2
shift;shift;
echo -n "$(basename $DEST)\t"
.venv/bin/python -t -m spacy evaluate "$@" $SRC $DEST | egrep --color=never "Time|NER|not" | tr '\n' '\t'
echo ""
