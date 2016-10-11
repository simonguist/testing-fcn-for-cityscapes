#---------------------------------------------------------
#Calculate mean pixel from dataset
#---------------------------------------------------------

from __future__ import division
from PIL import Image
import math
import os
import numpy as np

dataset_path='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/data/sbdd/dataset/img'

n=0
mean_sum=np.zeros(3)
for root, dirs, files in os.walk(dataset_path):
    for name in files:
       if ('leftImg8bit.png') and ('resized_2' in name) and (not '_cut_' in name):	#modify this line to control which images are found
           img=Image.open(root + '/' + name)ss
           in_ = np.array(img, dtype=np.float32)
           in_ = in_[:,:,::-1]
           mean_sum += in_.mean((0,1))
           n += 1

mean=mean_sum/n
print('evaluated ' + str(n) + ' images')
print('mean: ' + str(mean))
