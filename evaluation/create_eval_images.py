#---------------------------------------------------------
#Create evaluation images with label ids for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/export/home/sguist/caffe/python'
fcn_net='cityscapes-fcn8s-2x.caffemodel'	#the caffe model to be evaluated
path_imgs='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/data/cityscapes/leftImg8bit/val/'	#folder and subfolders will be searched
path_result='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/data/cityscapes/results/full_8/'     #folder where label images will be saved
pixel_mean=(71.60167789, 82.09696889, 72.30608881)      #mean pixel from the dataset the model was trained on

#---------------------------------------------------------


import sys
sys.path.insert(0, python_caffe_root)

from PIL import Image
import caffe
import numpy as np
import os
import setproctitle

setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
net = caffe.Net('deploy.prototxt', fcn_net, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)

eval_images=[]

#find all images
for root, dirs, files in os.walk(path_imgs):
    for name in files:
        if ('leftImg8bit' in name) and not ('resized_' in name) and not ("cut_" in name):  #modify this line to control which images are found
	     name_label=name.replace('leftImg8bit', 'gtFine_labelIds')
             eval_images.append((root+'/'+name, name, name_label))

n_images=len(eval_images)

#create label images
for idx in range(n_images):
    print(eval_images[idx][1])
    im = Image.open(eval_images[idx][0])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array(pixel_mean)
    in_ = in_.transpose((2,0,1))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    out_np=np.array(out, dtype=np.uint8)

    #replace train ids with label ids    
    l=[[0, 7], [1,8], [2,11], [3,12], [4,13], [5,17], [6,19], [7,20], [8,21], [9,22], [10,23], [11,24], [12,25], [13,26], [14,27],
    [15,28], [16,31], [17,32], [18,33]]
    
    out_np_new=np.zeros_like(out_np)  #initialize with unlabeld
    for i in l:
       out_np_new[out_np==i[0]]=i[1]

    #Save resulting array as image
    Image.fromarray(out_np_new).save(path_result+'eval_labels_' +  eval_images[idx][2], 'PNG')
