.PHONY: train

jupyter:
	screen .venv/bin/jupyter notebook --port=8300 --no-browser .

browser:
	screen .venv/bin/jupyter notebook --port=8300 .

S=.venv/bin/python -m spacy
D=data/UD_Russian-SynTagRus
F=$(date +"%m-%d-%y_%H-%M-%S")
M=data/model-$F

setup:
	test -d .venv || virtualenv .venv
	.venv/bin/pip3 install poetry
	CUDA_HOME=/usr/local/cuda .venv/bin/pip3 install --no-cache-dir thinc\<6.11
	#.venv/bin/pip3 uninstall -y thinc spacy
	poetry install || .venv/bin/poetry install
	.venv/bin/python3 -c "import thinc.neural.gpu_ops"

$D/ru_syntagrus-ud-train.json:
	test -d $D/ $S convert -m $D/ru_syntagrus-ud-train.conllu $D

$D/ru_syntagrus-ud-test.json:
	test -d $D/ $S convert -m $D/ru_syntagrus-ud-test.conllu $D

train: $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	OPENBLAS_NUM_THREADS=1 $S train -g 1 -G -N ru $M $D/ru_syntagrus-ud-train.json $D/ru_syntagrus-ud-test.json
	cp -r $M/* ru2/
