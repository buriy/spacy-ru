#!/bin/sh
CUDA=$1

.venv/bin/pip uninstall -y cupy cupy-$1
CUDA_ROOT=/usr/local/cuda .venv/bin/pip install --unstable-feature=resolver --no-cache-dir -U cupy\<8 cupy-$1\<8
