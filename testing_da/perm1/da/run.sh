#!/bin/sh
~/Gaze/bijcaffe/build/tools/caffe train -solver ./solver.prototxt
~/Gaze/bijcaffe/build/tools/caffe test -model ./cnn.prototxt -weights ./nets/_iter_10000.caffemodel
