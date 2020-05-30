#!/bin/sh
VER=$1
CUDA=$2

.venv/bin/pip uninstall -y spacy thinc cupy cupy-$2
CUDA_ROOT=/usr/local/cuda .venv/bin/pip install --unstable-feature=resolver --no-cache-dir -U spacy==$1 cupy cupy-$2 thinc==7.4.0
