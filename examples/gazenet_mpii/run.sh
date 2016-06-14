#!/bin/sh
~/Gaze/bijcaffe/build/tools/caffe train -solver ~/Gaze/bijcaffe/models/gazenet_mpii/solver.prototxt
#~/Software/caffe/build/tools/caffe train -solver ./solver.prototxt 2>&1 | tee train.log 
#~/Software/caffe/build/tools/caffe test -model ./train_test.prototxt -weights ./_iter_10000.caffemodel 2>&1 | tee test.log 
#~/Gaze/bijcaffe/build/tools/caffe test -model ./cnn.prototxt -weights ./_iter_10000.caffemodel                   
