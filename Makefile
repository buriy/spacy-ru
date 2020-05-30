.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

S:=.venv/bin/python -m spacy
D:=data/syntagrus
F:=$(shell date +"%m-%d-%y_%H-%M-%S")
V:=ru2
M:=data/model-${V}-${F}
#T:=$S train
T:=.venv/bin/python -m training.spacy_train

setup:
	test -d .venv || virtualenv --python=python3.6 .venv
	poetry install

setup_cuda91: setup
	./cuda.sh "<2.3" cuda91
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

setup_cuda100: setup
	./cuda.sh "<2.3" cuda100
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

data/syntagrus/ru_syntagrus-ud-train.connlu:
	mkdir -p data
	cd data; git clone https://github.com/UniversalDependencies/syntagrus

data/syntagrus/ru_syntagrus-ud-train.json:
	$S convert -m $D/ru_syntagrus-ud-train.conllu $D

data/syntagrus/ru_syntagrus-ud-test.json:
	$S convert -m $D/ru_syntagrus-ud-test.conllu $D

$D/ru_syntagrus-ud-dev.json:
	$S convert -m $D/ru_syntagrus-ud-dev.conllu $D

train-vec: data/syntagrus/ru_syntagrus-ud-train.json data/syntagrus/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $T -g 0 -cw 150 -b data/navec_hudlit.test.model -G -n 30 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	mkdir -p ru2_train/
	cp -r $M/model-final/* ru2_train/

train: data/syntagrus/ru_syntagrus-ud-train.json data/syntagrus/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $T -g 1 -cw 150 -G -n 30 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	mkdir -p ru2_train/
	cp -r $M/model-final/* ru2_train/

train-on-test: data/syntagrus/ru_syntagrus-ud-test.json data/syntagrus/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $T -g 1 -G -n 2 ru $M $D/ru_syntagrus-ud-test.json $D/ru_syntagrus-ud-test.json
	mkdir -p ru2_train_on_test/
	cp -r $M/model-final/* ru2_train_on_test/


train-on-dev: data/syntagrus/ru_syntagrus-ud-dev.json data/syntagrus/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $T -g 1 -G -n 2 ru $M $D/ru_syntagrus-ud-dev.json $D/ru_syntagrus-ud-test.json
	mkdir -p ru2_train_on_dev/
	cp -r $M/model-final/* ru2_train_on_dev/

train-ner: 
	echo $T -g 1  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	#mkdir -p $V/
	#cp -r $M/model-final/* $V/
