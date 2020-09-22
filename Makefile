.PHONY: setup train n nb

jupyter:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 --no-browser .

browser:
	PYTHONPATH=`pwd` screen .venv/bin/jupyter notebook --ip 0.0.0.0 --port=8881 .

# GPU id, -1 = train on CPU
GPU:=-1
CUDA:=91
CW:=96
WF:=-b data/models/navec_hudlit.model
OPTS:=
EPOCHS:=20
B:=syntagrus
N:=ru2_$B_${CW}
P:=tagger,parser,ner
S:=.venv/bin/python -u -m spacy
D:=data/$B
Dsyntagrus:=data/syntagrus
DT:=${D}/train.json
DD:=${Dsyntagrus}/dev.json
DE:=${Dsyntagrus}/test.json
G:=data/grameval
Gdev:=$G/GramEval2020-master/dataOpenTest
Gtrain:=$G/GramEval2020-master/dataTrain
F:=$(shell date +"%m-%d-%y_%H-%M-%S")
M:=data/models/${N}-${F}
T:=.venv/bin/python -u -m training.spacy_train

setup:
	test -d .venv || python3 -m venv .venv
	poetry install

setup_cuda: setup
	./cuda.sh cuda${CUDA}
	.venv/bin/python3 -c "import spacy;spacy.require_gpu()"

$G/GramEval2020-master:
	mkdir -p $G
	curl -L https://github.com/dialogue-evaluation/GramEval2020/archive/master.zip -o $G/master.zip
	#cp data/master.zip $G/master.zip
	cd $G; unzip master.zip

$G/poetry-dev.conllu: $G/GramEval2020-master
	cat ${Gdev}/GramEval2020-RuEval2017-Lenta-news-dev.conllu >$G/news-dev.conllu
	cat ${Gdev}/GramEval2020-GSD-wiki-dev.conllu >$G/wiki-dev.conllu
	cat ${Gdev}/GramEval2020-SynTagRus-dev.conllu >$G/fiction-dev.conllu
	cat ${Gdev}/GramEval2020-RuEval2017-social-dev.conllu >$G/social-dev.conllu
	cat ${Gdev}/GramEval2020-Taiga-poetry-dev.conllu >$G/poetry-dev.conllu

$G/poetry.conllu: $G/GramEval2020-master
	cat ${Gdev}/GramEval2020-RuEval2017-Lenta-news-dev.conllu ${Gtrain}/MorphoRuEval2017-Lenta-train.conllu >$G/news.conllu
	cat ${Gdev}/GramEval2020-GSD-wiki-dev.conllu ${Gtrain}/GramEval2020-GSD-train.conllu >$G/wiki.conllu
	cat ${Gdev}/GramEval2020-SynTagRus-dev.conllu ${Gtrain}/GramEval2020-SynTagRus-train-v2.conllu ${Gtrain}/MorphoRuEval2017-JZ-gold.conllu >$G/fiction.conllu
	cat ${Gdev}/GramEval2020-RuEval2017-social-dev.conllu ${Gtrain}/GramEval2020-Taiga-social-train.conllu ${Gtrain}/MorphoRuEval2017-VK-gold.conllu >$G/social.conllu
	cat ${Gdev}/GramEval2020-Taiga-poetry-dev.conllu ${Gtrain}/GramEval2020-Taiga-poetry-train.conllu >$G/poetry.conllu

$G/poetry-dev.json: $G/poetry-dev.conllu
	./convert.sh 1 $G/news-dev.conllu $G/news-dev.json
	./convert.sh 1 $G/wiki-dev.conllu $G/wiki-dev.json
	./convert.sh 1 $G/fiction-dev.conllu $G/fiction-dev.json
	./convert.sh 1 $G/social-dev.conllu $G/social-dev.json
	./convert.sh 1 $G/poetry-dev.conllu $G/poetry-dev.json

$G/poetry.json: $G/poetry.conllu
	./convert.sh 1 $G/news.conllu $G/news.json
	./convert.sh 1 $G/wiki.conllu $G/wiki.json
	./convert.sh 1 $G/fiction.conllu $G/fiction.json
	./convert.sh 1 $G/social.conllu $G/social.json
	./convert.sh 1 $G/poetry.conllu $G/poetry.json

$N/quality.txt: $G/poetry.json $G/poetry-dev.json
	echo "" > $@
	./eval.sh $N $G/news-dev.json -g ${GPU} >> $@
	./eval.sh $N $G/wiki-dev.json -g ${GPU} >> $@
	./eval.sh $N $G/fiction-dev.json -g ${GPU} >> $@
	./eval.sh $N $G/social-dev.json -g ${GPU} >> $@
	./eval.sh $N $G/poetry-dev.json -g ${GPU} >> $@
	./eval.sh $N $G/news.json -g ${GPU} >> $@
	./eval.sh $N $G/wiki.json -g ${GPU} >> $@
	./eval.sh $N $G/fiction.json -g ${GPU} >> $@
	./eval.sh $N $G/social.json -g ${GPU} >> $@
	./eval.sh $N $G/poetry.json -g ${GPU} >> $@

eval: $N/quality.txt
	cat $N/quality.txt | sed 's/fiction-dev/fic-dev/'

${Dsyntagrus}:
	git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git $@
	cp $@/ru_syntagrus-ud-train.conllu $@/train.conllu
	cp $@/ru_syntagrus-ud-test.conllu $@/test.conllu
	cp $@/ru_syntagrus-ud-dev.conllu $@/dev.conllu

$D:
	echo ""

${DT}: $D
	./convert.sh 10 $D/train.conllu $@

${DD}: ${Dsyntagrus}
	./convert.sh 1 ${Dsyntagrus}/test.conllu $@

${DE}: ${Dsyntagrus}
	./convert.sh 1 ${Dsyntagrus}/dev.conllu $@

data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar:
	curl https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar -o data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar

data/models/navec_hudlit.model: data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar
	.venv/bin/python -m training.navec2spacy -m data/navec/navec_hudlit_v1_12B_500K_300d_100q.tar -o data/models/navec_hudlit.model

$N/accuracy.txt: ${DT} ${DD} data/models/navec_hudlit.model
	mkdir -p $M
	OPENBLAS_NUM_THREADS=1 $T -p ${P} -R -g ${GPU} -cw ${CW} ${WF} ${OPTS} -G -n ${EPOCHS} ru $M ${DT} ${DD} | tee $M/accuracy.txt
	mkdir -p $N
	cp -r $M/model-final/* $N
	cp $M/accuracy.txt $@

train: $N/accuracy.txt

train-check: ${DE} ${DD} 
	OPENBLAS_NUM_THREADS=1 $T -g ${GPU} -G -n 2 ru $M ${DD} ${DE}
	mkdir -p ru2_train_on_test/
	cp -r $M/model-final/* ru2_train_on_test/
