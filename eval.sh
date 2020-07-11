#!/bin/sh
echo "$2:"
.venv/bin/python -t -m spacy evaluate -g 0 $1 $2 | egrep --color=never "UAS|LAS"
