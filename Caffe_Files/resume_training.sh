#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/Level_Alex_net/solver1.prototxt \
    --snapshot=models/Level_Alex_net/Level_Alex_net_iter_10000.solverstate \
    $@
