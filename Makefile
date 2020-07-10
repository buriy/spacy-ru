.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

# GPU id, -1 = train on CPU
G:=0

S:=.venv/bin/python -u -m spacy
D:=data/syntagrus
G:=data/grameval
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

$G/GramEval2020-master:
	mkdir -p $G
	#wget https://github.com/dialogue-evaluation/GramEval2020/archive/master.zip -O $G/master.zip
	cp data/master.zip $G/master.zip
	cd $G; unzip master.zip

$G/fiction.json: $G/GramEval2020-master
	./convert.sh $G/GramEval2020-master/dataOpenTest/GramEval2020-RuEval2017-Lenta-news-dev.conllu >$G/news.json
	./convert.sh $G/GramEval2020-master/dataOpenTest/GramEval2020-GSD-wiki-dev.conllu >$G/wiki.json
	./convert.sh $G/GramEval2020-master/dataOpenTest/GramEval2020-SynTagRus-dev.conllu >$G/fiction.json
	./convert.sh $G/GramEval2020-master/dataOpenTest/GramEval2020-RuEval2017-social-dev.conllu >$G/social.json
	./convert.sh $G/GramEval2020-master/dataOpenTest/GramEval2020-Taiga-poetry-dev.conllu >$G/poetry.json

ru2_raw/quality.txt:
	echo "" > $@
	./eval.sh ru2_raw $G/news.json >> $@
	./eval.sh ru2_raw $G/wiki.json >> $@
	./eval.sh ru2_raw $G/fiction.json >> $@
	./eval.sh ru2_raw $G/social.json >> $@
	./eval.sh ru2_raw $G/poetry.json >> $@
	cat $@

ru2_syntagrus/quality.txt:
	echo "" > $@
	./eval.sh ru2_syntagrus $G/news.json >> $@
	./eval.sh ru2_syntagrus $G/wiki.json >> $@
	./eval.sh ru2_syntagrus $G/fiction.json >> $@
	./eval.sh ru2_syntagrus $G/social.json >> $@
	./eval.sh ru2_syntagrus $G/poetry.json >> $@
	cat $@

$D:
	git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git $D

$D/train.json: $D
	./convert.sh $D/ru_syntagrus-ud-train.conllu $D/train.json

$D/test.json: $D
	./convert.sh $D/ru_syntagrus-ud-test.conllu $D/test.json

$D/dev.json: $D
	./convert.sh $D/ru_syntagrus-ud-dev.conllu $D/dev.json

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
