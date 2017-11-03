image-classification-segmentation

* background

* problem

    - link to VOC2012
    
* approach

    - link to the paper on fcn
 
* implementation

    - keras
    - link to author's git repo (caffe)
    - link to D's git repo (tensorflow)
    

* More theoretical description: read final report

* conda and virtual environment

    - git clone https://github.com/Charlio/image-classification-segmentation.git
    - download and install conda: https://conda.io/docs/user-guide/install/download.html
    - create conda environment, in the git repo: conda create --name keras --file keras-env.yml
    - activate the environment: source activate keras or activate keras
    

* dataset 

    - Pascal VOC2012 image dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
    - click the link: Download the training/validation data (2GB tar file)
    - then move the extracted folder VOCdevkit to your repo directory
    - run data.py to generate training data files .npy

* model weights
    
    - If you have trouch downloading VGG16 in the notebook when calling keras.applications.vgg16.VGG16, then use the following link to manually download it, and put in ~/.keras/models/. VGG16 weights (~ 500m) is needed for training FCN32, if you do not plan to train FCN32, then you don't need to download it.
    - [VGG16](https://www.dropbox.com/home/fcn-model-weights?preview=vgg16_weights_tf_dim_ordering_tf_kernels.h5) 
    - Download the following weights of FCN32(~1.6G), and put it in your repo/saved_models/ . The weights is needed for prediction demonstration, as well as training FCN16. If you do not want to train FCN16 or predict with FCN32, then you don't need to download it.
    - [FCN32](https://www.dropbox.com/home/fcn-model-weights?preview=fcn32_7761.h5)
    - Download the following weights of FCN16(~1.7G), and put it in your repo/saved_models/ . The weights is needed for prediction demonstration with FCN16.
    - [FCN16](https://www.dropbox.com/home/fcn-model-weights?preview=fcn16_7835.h5) 

* data exploration and model prediction demonstration

    - data-exploration-and-prediction-demo.ipynb

* model review and training

    - model-review-and-training.ipynb
