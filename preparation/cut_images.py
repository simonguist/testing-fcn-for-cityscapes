#---------------------------------------------------------
#Cut images in half
#---------------------------------------------------------

from __future__ import division
from PIL import Image
import math
import os

fcn_root='/export/home/sguist/fcn/fcn.berkeleyvision.org-master/'
path=fcn_root+'data/cityscapes/'


for root, dirs, files in os.walk(path):
    for name in files:
        if ('leftImg8bit.png' in name or 'gtFine_labelIds.png' in name or 'gtFine_labelTrainIds.png' in name) and (not '_cut_' in name) and (not os.path.isfile(root + '/' + 'left_cut_' + name)):  #modify this line to control which images are found
	   print(name)
 	   img=Image.open(root + '/' + name)	
	   w, h = img.size
	   filename, extension = name.split(".")
	   img.crop((0, 0, int(w*0.5), int(h)).save(root + '/' + 'left_cut_' + name)
	   img.crop((int(w*0.5), 0, int(w), int(h))).save(root + '/' + 'right_cut_' + name)
