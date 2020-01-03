.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

S:=.venv/bin/python -m spacy
D:=data/UD_Russian-SynTagRus
F:=$(shell date +"%m-%d-%y_%H-%M-%S")
V:=ru2
M:=data/model-${V}-${F}

setup:
	test -d .venv || virtualenv --python=python3.6 .venv
	poetry install

setup_cuda91: setup
	./cuda.sh "<2.2" cuda91
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

setup_cuda100: setup
	./cuda.sh "<2.2" cuda100
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

setup_cuda101: setup
	./cuda.sh "<2.2" cuda101
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.connlu:
	mkdir -p data
	cd data; git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus

data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.json:
	$S convert -m $D/ru_syntagrus-ud-train.conllu $D

data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.json:
	$S convert -m $D/ru_syntagrus-ud-test.conllu $D

train: data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.json data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $S train -g 0  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	mkdir -p $V/
	cp -r $M/model-final/* $V/

train-ner: 
	echo $S train -g 0  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	#mkdir -p $V/
	#cp -r $M/model-final/* $V/
