#---------------------------------------------------------
# Test multiple models

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/export/home/sguist/caffe/python'
fcn_root='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/'				#path containing cityscapes_layers.py and score.py
weights = [fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_5000.caffemodel',			#all models to be evaluated
	   fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_10000.caffemodel', 
	   fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_15000.caffemodel', 
           fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_20000.caffemodel',
	   fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_25000.caffemodel',
	   fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_30000.caffemodel',
	   fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_35000.caffemodel',
           fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_40000.caffemodel',
           fcn_root + 'cityscapes-fcn8s/snapshot/train_iter_45000.caffemodel',
	   fcn_root + 'cityscapes-fcn8s/cityscapes-fcn8s-2x.caffemodel' 
	]                     	
n_steps=50000											#number of training iterations (one iteration = one random image from the training set)
final_model_name='cityscapes-fcn16s-2x.caffemodel'
n_val=500;											#number of images to evaluate on (max number of images in dataset)

#---------------------------------------------------------



import sys
sys.path.insert(0, python_caffe_root)
sys.path.insert(0, fcn_root)

import caffe
import surgery, score
import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
caffe.set_mode_gpu()
caffe.set_device(0)

#do evaluations
for path in weights:
    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(path)
    loss=score.seg_tests_and_get_loss(solver, False, n_val, layer='score')
