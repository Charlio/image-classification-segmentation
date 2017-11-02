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

    - VGG16, FCN32, FCN16 

* data exploration and model prediction demonstration

    - data-exploration-and-prediction-demo.ipynb

* model review and training

    - model-review-and-training.ipynb
