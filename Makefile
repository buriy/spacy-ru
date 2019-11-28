.PHONY: setup train n nb

jupyter:
	screen .venv/bin/jupyter notebook --port=8881 --no-browser .

browser:
	screen .venv/bin/jupyter notebook --port=8881 .

S:=.venv/bin/python -m spacy
D:=data/UD_Russian-SynTagRus
F:=$(shell date +"%m-%d-%y_%H-%M-%S")
V:=ru2
M:=data/model-${V}-${F}

setup:
	test -d .venv || virtualenv --python=python3.6 .venv
	poetry install

gpu21:
	.venv/bin/pip3 uninstall -y thinc
	CUDA_HOME=/usr/local/cuda .venv/bin/pip3 install --upgrade --no-cache-dir cupy
	CUDA_HOME=/usr/local/cuda .venv/bin/pip3 install --upgrade --no-cache-dir cupy-cuda100
	CUDA_HOME=/usr/local/cuda .venv/bin/pip3 install --no-cache-dir thinc==7.0.8
	poetry install
	.venv/bin/python3 -c "import thinc.neural.gpu_ops"


data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.connlu:
	mkdir -p data
	cd data; git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus

data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.json:
	$S convert -m $D/ru_syntagrus-ud-train.conllu $D

data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.json:
	$S convert -m $D/ru_syntagrus-ud-test.conllu $D

train: data/UD_Russian-SynTagRus/ru_syntagrus-ud-train.json data/UD_Russian-SynTagRus/ru_syntagrus-ud-test.json
	$S train -g 0  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	mkdir -p $V/
	cp -r $M/model-final/* $V/

train-ner: 
	echo $S train -g 0  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	#mkdir -p $V/
	#cp -r $M/model-final/* $V/
