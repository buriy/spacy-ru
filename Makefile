.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

# GPU id, -1 = train on CPU
GPU:=0
S:=.venv/bin/python -u -m spacy
D:=data/syntagrus
G:=data/grameval
Gdev:=$G/GramEval2020-master/dataOpenTest
Gtrain:=$G/GramEval2020-master/dataTrain
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

$G/fiction.conllu:
	cat ${Gdev}/GramEval2020-RuEval2017-Lenta-news-dev.conllu ${Gtrain}/MorphoRuEval2017-Lenta-train.conllu >$G/news.conllu
	cat ${Gdev}/GramEval2020-GSD-wiki-dev.conllu ${Gtrain}/GramEval2020-GSD-train.conllu >$G/wiki.conllu
	cat ${Gdev}/GramEval2020-SynTagRus-dev.conllu ${Gtrain}/GramEval2020-SynTagRus-train-v2.conllu ${Gtrain}/MorphoRuEval2017-JZ-gold.conllu >$G/fiction.conllu
	cat ${Gdev}/GramEval2020-RuEval2017-social-dev.conllu ${Gtrain}/GramEval2020-Taiga-social-train.conllu ${Gtrain}/MorphoRuEval2017-VK-gold.conllu >$G/social.conllu
	cat ${Gdev}/GramEval2020-Taiga-poetry-dev.conllu ${Gtrain}/GramEval2020-Taiga-poetry-train.conllu >$G/poetry.conllu

$G/fiction.json: $G/fiction.conllu
	./convert.sh $G/news.conllu $G/news.json
	./convert.sh $G/wiki.conllu $G/wiki.json
	./convert.sh $G/fiction.conllu $G/fiction.json
	./convert.sh $G/social.conllu $G/social.json
	./convert.sh $G/poetry.conllu $G/poetry.json

ru2_raw/quality.txt: ru2_raw
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

eval: ru2_syntagrus/quality.txt

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

ru2_syntagrus: $D/train.json $D/test.json data/models/navec_hudlit.model
	#rm -rf $@
	mkdir -p $@
	OPENBLAS_NUM_THREADS=1 $T -g ${GPU} -cw 150 -b data/models/navec_hudlit.model -G -n 30 ru $M $D/train.json $D/test.json | tee $@/accuracy.txt -
	cp -r $M/model-final/* $Y/

ru2_raw: $D/train.json $D/test.json
	OPENBLAS_NUM_THREADS=1 $T -g ${GPU} -cw 150 -G -n 30 ru $M $D/train.json $D/test.json
	mkdir -p $@
	cp -r $M/model-final/* $@

train-check: $D/dev.json $D/test.json 
	OPENBLAS_NUM_THREADS=1 $T -g ${GPU} -G -n 2 ru $M $D/dev.json $D/test.json
	mkdir -p ru2_train_on_dev/
	cp -r $M/model-final/* ru2_train_on_dev/

train-ner: 
	echo $T -g ${GPU} -G -n 20 ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	#mkdir -p $V/
	#cp -r $M/model-final/* $V/

retrain_syntagrus:
	rm -rf ru2_syntagrus
	make ru2_syntagrus