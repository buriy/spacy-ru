.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

# GPU id, -1 = train on CPU
G:=0

S:=.venv/bin/python -m spacy
D:=data/syntagrus
F:=$(shell date +"%m-%d-%y_%H-%M-%S")
V:=ru2
M:=data/models/${V}-${F}
T:=.venv/bin/python -u -m training.spacy_train

setup:
	test -d .venv || python3 -m venv .venv
	poetry install

setup_cuda91: setup
	./cuda.sh "<2.4" cuda91
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

setup_cuda100: setup
	./cuda.sh "<2.4" cuda100
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

$D:
	git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git $D

$D/train.conllu: $D
	sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $D/ru_syntagrus-ud-train.conllu >$D/train~.conllu
	mv $D/train~.conllu $D/train.conllu

$D/test.conllu: $D
	sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $D/ru_syntagrus-ud-test.conllu >$D/test~.conllu
	mv $D/test~.conllu $D/test.conllu

$D/dev.conllu: $D
	sed 's/Variant=/StyleVariant=/g; s/=1/=First/g; s/=2/=Second/g; s/=3/=Third/g;' $D/ru_syntagrus-ud-dev.conllu >$D/dev~.conllu
	mv $D/dev~.conllu $D/dev.conllu

$D/train.json: $D/train.conllu
	$S convert -m $D/train.conllu $D

$D/test.json: $D/test.conllu
	$S convert -m $D/test.conllu $D

$D/dev.json: $D/dev.conllu
	$S convert -m $D/dev.conllu $D

data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar:
	echo "Please download yourself"

data/models/navec_hudlit.model: data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar
	.venv/bin/python -m training.navec2spacy -m data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar -o data/models/navec_hudlit.model

train-syntagrus: $D/train.json $D/test.json data/models/navec_hudlit.model
	Y = ru2_syntagrus_navec
	rm -rf $Y/
	mkdir -p $Y/
	OPENBLAS_NUM_THREADS=1 $T -g $G -cw 150 -b data/models/navec_hudlit.model -G -n 30 ru $M $D/train.json $D/test.json | tee ru2_syntagrus/accuracy.txt -
	cp -r $M/model-final/* $Y/

train-syntagrus-raw: $D/train.json $D/test.json
	Y = ru2_syntagrus_raw
	OPENBLAS_NUM_THREADS=1 $T -g $G -cw 150 -G -n 30 ru $M $D/train.json $D/test.json
	mkdir -p ru2_syntagrus_raw/
	cp -r $M/model-final/* ru2_syntagrus_raw/

train-check: $D/dev.json $D/test.json 
	OPENBLAS_NUM_THREADS=1 $T -g $G -G -n 2 ru $M $D/dev.json $D/test.json
	mkdir -p ru2_train_on_dev/
	cp -r $M/model-final/* ru2_train_on_dev/

train-ner: 
	echo $T -g $G  -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	#mkdir -p $V/
	#cp -r $M/model-final/* $V/
