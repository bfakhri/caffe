#!/bin/sh
~/Gaze/bijcaffe/build/tools/caffe train -solver ~/Gaze/bijcaffe/testing_da/t1/solver.prototxt
#~/Software/caffe/build/tools/caffe train -solver ./solver.prototxt 2>&1 | tee train.log 
#~/Software/caffe/build/tools/caffe test -model ./train_test.prototxt -weights ./_iter_10000.caffemodel 2>&1 | tee test.log 
~/Gaze/bijcaffe/build/tools/caffe test -model ~/Gaze/bijcaffe/testing_da/t1/cnn.prototxt -weights ~/Gaze/bijcaffe/testing_da/t1/trained_nets/_iter_10000.caffemodel
#~/Gaze/bijcaffe/build/tools/caffe test -model ~/Gaze/bijcaffe/models/gazenet_mmd/cnn.prototxt -weights ./_iter_10000.caffemodel
