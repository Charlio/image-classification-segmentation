"""
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def cnn10(classes=10):
    """Instantiates the cnn16 architecture.
    # Arguments
        classes: optional number of classes to classify images
            into, only to be specified if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
        
    x = Sequential()
    
    # Block 1
    x.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', name='block1_conv1'))
    x.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    x.add(Dropout(0.5))

    # Block 2
    x.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    x.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    x.add(Dropout(0.5))
    
    # Block 3
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    x.add(Dropout(0.5))
    
    # Dense block
    x.add(Flatten(name='flatten'))
    x.add(Dense(256, activation='relu', name='fc1'))
    x.add(Dropout(0.5))
    x.add(Dense(256, activation='relu', name='fc2'))
    x.add(Dropout(0.5))
    x.add(Dense(classes, activation='softmax', name='predictions'))

    return x