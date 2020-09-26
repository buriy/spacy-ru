#!/bin/bash
echo "Args:" "$@"
n=$1
shift;
sources=${@:1:$#-1}
target=${@:$#}
mkdir $target 2>/dev/null
for s in $sources; do
    fn="${s##*/}"
    echo 'Converting' $s
    # ./convert.sh 1 $2 $3
    sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g; s/Tag=//g; s/|SpaceAfter=No//g' $s >>/tmp/$fn.$$.conllu
    .venv/bin/python -u -m spacy convert -n $n -m /tmp/$fn.$$.conllu $target
done
echo "OK"
touch $target/.done
