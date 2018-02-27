#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/Level_Net
DATA=data/Level_Net_Data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/Level_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
