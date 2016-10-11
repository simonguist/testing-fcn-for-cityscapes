# testing-fcn-for-cityscapes-dataset
Testing fully convolutional networks for semantic segmentation with caffe for the cityscapes dataset 

## How to get started
- Download the cityscapes dataset and the vgg-16-layer net
- Modify the images in the dataset with cut_images.py or downscale_images.py for less resource demanding training and evaluation
- Run net_32.py to create the stride 32 pixel stride net
- Modify the paths in train.txt and val.txt (first line: path to training/validation images, second line: path to annotations)
- Run solve_start.py to start training
- Run evaluate_models.py to evaluate your model or create_eval_images.py	to create images with pixel label ids


## Sources

### Fully Convolutional Models for Semantic Segmentation:
Shelhamer, Evan, Jonathon Long, and Trevor Darrell. "Fully Convolutional Networks for
Semantic Segmentation." PAMI, 2016, URL http://fcn.berkeleyvision.org

### Cityscapes Dataset (Semantic Understanding of Urban Street Scenes):
Cordts, Marius, et al. "The cityscapes dataset." CVPR Workshop on The Future of Datasets
in Vision. 2015, URL https://www.cityscapes-dataset.com

### Caffe framework:
Shelhamer, Evan, Jonathon Long, and Trevor Darrell. "Fully Convolutional Networks for
Semantic Segmentation." PAMI, 2016, URL http://caffe.berkeleyvision.org
