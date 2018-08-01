#!/bin/sh

cd ../
## MODIFY PATH for YOUR SETTING
CAFFE_DIR=/Users/camlin_z/Data/Project/caffe-68landmark
CONFIG_DIR=${CAFFE_DIR}/landmark_detec
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe
DEV_ID=0

sudo ${CAFFE_BIN} train \
-solver=${CONFIG_DIR}/solver.prototxt \
-weights=${CONFIG_DIR}/init.caffemodel \
# -gpu=${DEV_ID} \
2>&1 | tee ${CONFIG_DIR}/train.log
