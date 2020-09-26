#!/bin/bash
#echo "Args:" "$@"
n=$1
shift;
sources=${@:1:$#-1}
target=${@:$#}
mkdir $target 2>/dev/null
for s in $sources; do
    ./convert.sh $n $s $target
done
echo "All files converted."
touch $target/.done
