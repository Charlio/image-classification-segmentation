# Machine Learning Engineer Nanodegree
## Capstone Project -- Fully Convolutional Network for Image Segmentation
Charlio Xu
November 2nd, 2017

## I. Definition


### Project Overview

Computer vision is a popular and fascinating field for deep learning.

- Image classification, object detection, localization, segmentation
- convolutional neural network, fully convolutional network
- paper link: [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- show one pair of images and masks

### Problem Statement

The goal of this project is to implement the fully convolutional networks FCN32 and FCN16 fine tuned on VGG16 to give pixel-wise classification of images in order to detailed information about the localization and shape of common objects in the images. Tasks involved in the project are the following:
    1. Download and preprocess PASCAL VOC2012 data
    2. Design and implement FCN32 and FCN16 models in keras
    3. Train FCN32 and FCN16 with VOC segmentation data
    4. Demonstrate the predictions from FCN32 and FCN16


### Metrics

Dice coefficient is the common choice in object recognition and localization tasks. 

- description
- definition
- corresponding loss function


## II. Analysis

### Data Exploration

introduction to PASCAL VOC2012 dataset, content, volume, image size, challenges, link, image and seg demo

    - segmentation stats: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#tbl:segstats


### Exploratory Visualization

show several image and mask pairs

### Algorithms and Techniques

- intro to cnn
- intro to fcn
- idea of combining shallow layers with deep layers
- deconvolutionar layers
- weights transfer VGG16 to FCN32, FCN32 to FCN16

### Benchmark

- Berkeley link: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/


## III. Methodology

### Data Preprocessing

- original images: mean, std
- segs: transform into binary masks
- resize

### Implementation

- training of FCN32: weights from VGG16, data, optimizer, loss, metric, lr, etc
- computational graph
- training of FCN16: weights from FCN16, etc
- computational graph
- prediction function

### Refinement

- train for longer time on cloud with gpu
- lr refine


## IV. Results

### Model Evaluation and Validation

- validation set
- dice coefficient
- performance on test images: localization; coarse shape, long thin part

### Justification

- hardware config
- training config

## V. Conclusion

### Free-Form Visualization

- show prediction masks and images

### Reflection

- steps summary
- bottleneck, difficulties
- deconv vs bilinear upsampling

### Improvement

- more prediction classes instead of two
- train longer time
- link to orginal author (caffe), D's blog (tensorflow)
