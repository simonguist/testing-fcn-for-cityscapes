
Testing fully convolutional networks for semantic segmentation with caffe for the cityscapes dataset 

## How to get started
- Download the cityscapes dataset and the vgg-16-layer net
- Modify the images in the dataset with cut_images.py or downscale_images.py for less resource demanding training and evaluation
- Create the 32 pixel stride net with net_32.py 
- Modify the paths in train.txt and val.txt (first line: path to training/validation images, second line: path to annotations)
- Start training with solve_start.py 
- Run evaluate_models.py to evaluate your model or create_eval_images.py to create images with pixel label ids


## Sources

### Fully Convolutional Models for Semantic Segmentation:
Shelhamer, Evan, Jonathon Long, and Trevor Darrell. "Fully Convolutional Networks for Semantic Segmentation." PAMI, 2016,
URL http://fcn.berkeleyvision.org

### Cityscapes Dataset (Semantic Understanding of Urban Street Scenes):
Cordts, Marius, et al. "The cityscapes dataset." CVPR Workshop on The Future of Datasets in Vision. 2015,
URL https://www.cityscapes-dataset.com

### Caffe Deep Learning Framework:
Jia, Yangqing, et al. "Caffe: Convolutional architecture for fast feature embedding." Proceedings of the 22nd ACM international conference on Multimedia. ACM, 2014,
URL http://caffe.berkeleyvision.org
