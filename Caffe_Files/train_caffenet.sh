#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/Level_Alex_net/solver1.prototxt 2>&1 | tee models/Level_Alex_net/Train.log $@
