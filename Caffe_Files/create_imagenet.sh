#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=examples/Level_Net
DATA=/home/iot_lab_dnn/caffe/data/Level_Net_Data
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/iot_lab_dnn/caffe/data/Level_Net_Data/Training/
VAL_DATA_ROOT=/home/iot_lab_dnn/caffe/data/Level_Net_Data/Validate/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
DEPTH=true
if $RESIZE; then
  RESIZE_HEIGHT=28
  RESIZE_WIDTH=28
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi



if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --gray=$DEPTH\
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/lev_det_train.txt \
    $EXAMPLE/Level_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --gray=$DEPTH\
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/lev_det_test.txt \
    $EXAMPLE/Level_val_lmdb

echo "Done."
