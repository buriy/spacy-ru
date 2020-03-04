#!/bin/sh
VER=$1
CUDA=$2

#.venv/bin/pip uninstall spacy
.venv/bin/pip install -U spacy[$CUDA]$VER
.venv/bin/pip uninstall -y thinc
CUDA_ROOT=/usr/local/cuda .venv/bin/pip install --no-cache-dir thinc==7.0.8
