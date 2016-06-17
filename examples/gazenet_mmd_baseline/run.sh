#!/bin/sh
~/Gaze/bijcaffe/build/tools/caffe train -solver ~/Gaze/bijcaffe/models/gazenet_mmd/solver.prototxt
#~/Software/caffe/build/tools/caffe train -solver ./solver.prototxt 2>&1 | tee train.log 
#~/Software/caffe/build/tools/caffe test -model ./train_test.prototxt -weights ./_iter_10000.caffemodel 2>&1 | tee test.log 
~/Gaze/bijcaffe/build/tools/caffe test -model ~/Gaze/bijcaffe/models/gazenet_mmd/cnn.prototxt -weights ~/Gaze/bijcaffe/models/gazenet_mmd/trained_nets/_iter_10000.caffemodel
#~/Gaze/bijcaffe/build/tools/caffe test -model ~/Gaze/bijcaffe/models/gazenet_mmd/cnn.prototxt -weights ./_iter_10000.caffemodel
