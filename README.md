image-classification-segmentation

* For detailed description, please read the final report in the final_report folder

* conda and virtual environment

    - git clone https://github.com/Charlio/image-classification-segmentation.git
    - download and install conda: https://conda.io/docs/user-guide/install/download.html
    - create conda environment, in the git repo: 
            
            conda create --name keras --file keras-env.yml
            
    - activate the environment: 
            
            source activate keras 
            
            activate keras
    
* dataset 

    - Pascal VOC2012 image dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
    - click the link: Download the training/validation data (2GB tar file)
    - then move the extracted folder VOCdevkit to your repo directory
    - run data.py to generate training data files .npy:
            
            python data.py

* model weights
    
    - If you have trouch downloading VGG16 in the notebook when calling keras.applications.vgg16.VGG16, then use the following link to manually download it, and put in 
    
            ~/.keras/models/. 
            
    VGG16 weights (~ 500MB) is needed for training FCN32, if you do not plan to train FCN32, then you don't need to download it.
    
    - [VGG16](https://www.dropbox.com/s/jl6m0vk42c3sogf/vgg16_weights_tf_dim_ordering_tf_kernels.h5?dl=0) 
    
    - Download the following weights of FCN32(~1.5GB), and put it in your repo/saved_models/ . The weights is needed for prediction demonstration, as well as training FCN16. If you do not want to train FCN16 or predict with FCN32, then you don't need to download it.
    
    - [FCN32](https://www.dropbox.com/s/ztnlouvsaelcjsg/fcn32_7761.h5?dl=0)
    
    - Download the following weights of FCN16(~1.5GB), and put it in your repo/saved_models/ . The weights is needed for prediction demonstration with FCN16.
    
    - [FCN16](https://www.dropbox.com/s/0ybm110bvrt36ti/fcn16_7835.h5?dl=0) 

* data exploration and model prediction demonstration

    - open data-exploration-and-prediction-demo.ipynb to quickly go through the data set and see the prediction resulst from pre-trained models to have the first-sense about what this project does in the end.

* model review and training

    - model-review-and-training.ipynb
    
* FCN32 and FCN16 model definitions is given in models/
