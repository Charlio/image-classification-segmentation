from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Cropping2D

def FCN16(classes=2):
    
    x = Sequential()
    # Block 1
    x.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu', padding='same', name='block1_conv1'))
    x.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    x.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    x.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    x.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    x.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    x.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    x.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # Fully convolutional layers
    x.add(Conv2D(4096, (7, 7), activation='relu', padding='same', name='fcn1'))
    x.add(Dropout(0.5))
    x.add(Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn2'))
    x.add(Dropout(0.5))
    x.add(Conv2D(classes, (1, 1), padding='valid', name='fcn3'))

    x.add(Conv2DTranspose(classes, (64, 64), padding='valid', strides=(32, 32), name='deconv1'))
    x.add(Cropping2D(cropping=((16, 16), (16, 16))))
    x.add(Dense(classes, activation='softmax', name='predictions'))
    
    return x