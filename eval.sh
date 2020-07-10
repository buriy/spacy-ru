#!/bin/sh
echo "$2:"
.venv/bin/python -m spacy evaluate -g 0 $1 $2 | egrep --color=never "UAS|LAS"
