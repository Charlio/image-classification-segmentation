image-classification-segmentation

* install conda and virtual environment from env file

* download Pascal VOC2012 image dataset from
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

    - click the link: Download the training/validation data (2GB tar file)

    - then move the extracted folder VOCdevkit to your repo directory

    - run data.py to generate training data files npy

* download weights of VGG16, FCN32, FCN16, and put in saved_models

* data-exploration-and-prediction-demo.ipynb

* model-review-and-training.ipynb

* tasks to do
    - upload fcn32 (1.6G), fcn16 (1.7G) weights
    - first draft of final report
    - further parameter tuning
    - run on cloud and use more gpus
    - augment data