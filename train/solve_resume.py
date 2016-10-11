#---------------------------------------------------------
# Resume training a model for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/export/home/sguist/caffe/python'
fcn_root='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/'				#path containing cityscapes_layers.py and surgery.py
weights = fcn_root + 'cityscapes-fcn8s/cityscapes-fcn8s-2x.caffemodel'                     	#previous model for initialization
n_steps=50000											#number of training iterations (one iteration = one random image from the training set)
final_model_name='cityscapes-fcn16s-2x.caffemodel'

#---------------------------------------------------------



import sys
sys.path.insert(0, python_caffe_root)
sys.path.insert(0, fcn_root)


import caffe
import surgery
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))


# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver.step(n_steps)

solver.net.save(final_model_name)
