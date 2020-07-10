#!/bin/sh
sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $1 >$1.~
.venv/bin/python -t -m spacy -m convert -m $1.~ | tee >$2
