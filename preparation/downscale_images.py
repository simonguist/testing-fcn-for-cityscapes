#---------------------------------------------------------
# Downscale images
#---------------------------------------------------------


from __future__ import division
from PIL import Image
import math
import os

fcn_root='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/'
path=fcn_root+'data/cityscapes'


for root, dirs, files in os.walk(path):
    for name in files:
        if ('leftImg8bit.png' in name or 'gtFine_labelIds.png' in name or 'gtFine_labelTrainIds.png' in name) and (not '_cut_' in name) and (not os.path.isfile(root + '/' + 'resized_2_' + name) and (not 'resized_' in name)):   #modify this line to control which images are downscaled
 	   print(name)
	   img=Image.open(root + '/' + name)	
	   w, h = img.size
	   filename, extension = name.split(".")
	   img.resize((int(w/2), int(h/2))).save(root + '/' + 'resized_2_' + name)
