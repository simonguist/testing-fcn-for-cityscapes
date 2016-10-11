#---------------------------------------------------------
#Python Layer for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#---------------------------------------------------------

import caffe
import numpy as np
from PIL import Image
import os

import random


class CityscapesSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - cityscapes_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(cityscapes_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """

        # config
        params = eval(self.param_str)
        self.cityscapes_dir = params['cityscapes_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.cityscapes_dir,
                self.split)

        self.folders = open(split_f, 'r').read().splitlines()
	
	print(self.folders)

	self.indices=[]
	
	path_imgs=self.folders[0];
	for root, dirs, files in os.walk(path_imgs):
		for name in files:
			if ('leftImg8bit' in name) and ('resized_2' in name):				#modify this line to control which images are found
				name_label=name.replace('leftImg8bit', 'gtFine_labelTrainIds')
				root_label=root.replace(self.folders[0], self.folders[1])
				self.indices.append((root+'/'+name, root_label+'/'+name_label))


        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
	im = Image.open(idx[0])        
	in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
	im = Image.open(idx[1])        
	label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
