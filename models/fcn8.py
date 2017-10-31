"""
"""
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Cropping2D

def fcn8(classes=10):
    
    x = Sequential()
    
    # Block 1
    x.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', name='block1_conv1'))
    x.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    x.add(Dropout(0.5)) # 32, 16, 16, 3

    # Block 2
    x.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    x.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    x.add(Dropout(0.5)) # 64, 8, 8, 3
    
    # Block 3
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    x.add(Dropout(0.5)) # 128, 4, 4, 3
    
    # Continue to use convolutional layers instead of dense layers in cnn10
    x.add(Conv2D(256, (4, 4), activation='relu', padding='same', name='fcn1'))
    x.add(Dropout(0.5)) # 256, 4, 4, 3
    x.add(Conv2D(256, (1, 1), activation='relu', padding='same', name='fcn2'))
    x.add(Dropout(0.5)) # 256, 4, 4, 3

    x.add(Conv2D(classes, (1, 1), padding='valid', name='fc_score'))  # classes, 4, 4, 3  
    x.add(Conv2DTranspose(classes, (16, 16), strides=(8, 8), padding='valid', name='fc_upsampling'))  # classes, 40, 40, 3
    x.add(Cropping2D(cropping=((4, 4), (4, 4)))) # classes, 32, 32, 3
    
    return x
