#!/bin/sh
# Rebuild the PTX of src/*.cu (needs docker and the NGC image Coral pins).
set -e
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/w" -w /w nvcr.io/nvidia/pytorch:25.04-py3 \
  nvcc -O3 -std=c++17 --ptx -arch=sm_121a src/loop.cu -o ptx/loop.ptx
