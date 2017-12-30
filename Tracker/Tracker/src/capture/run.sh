#!/bin/bash

#export DIRECTCONV=1
export BYPASSACL=0x114c
export LOGACL=0x0
export OPENBLAS_NUM_THREADS=1



PROTOTXT=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/shallow-squeezenet_128.prototxt
#PROTOTXT=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/goturnDeploy_2_fc.prototxt
#PROTOTXT=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/goturnDeploy.prototxt
#PROTOTXT=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/tracker.prototxt

MODEL=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/models/pretrained_model/shallow-squeezenet_128_iter_250000.caffemodel
#MODEL=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/models/pretrained_model/goturn_iter_2_fc_580000.caffemodel
#MODEL=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/models/pretrained_model/goturn_iter_470000.caffemodel
#MODEL=/home/firefly/TrackerGoturn-master/TrackerGoturnDemo/nets/models/pretrained_model/tracker.caffemodel
GPU_ID=3


export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/opencv/lib:/home/firefly/trax/build:/usr/local/arm64/lib:/home/firefly/CaffeOnACL/distribute/lib:/home/firefly/ComputeLibrary/build/:$(LD_LIBRARY_PATH)


taskset -c 4 ./capture_tracker $PROTOTXT $MODEL $GPU_ID
