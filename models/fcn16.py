from keras.models import Model
from keras.layers import Dropout, Dense, Add, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Cropping2D

def FCN16(classes=2):
    
    inputs = Input(shape=(224, 224, 3))
    # Block 1
    block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

    # Block 4
    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

    # Block 5
    block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)
    
    # Fully convolutional layers
    fcn1 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fcn1')(block5_pool)
    drop1 = Dropout(0.5)(fcn1)
    fcn2 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn2')(drop1)
    drop2 = Dropout(0.5)(fcn2)
    fcn3 = Conv2D(classes, (1, 1), padding='valid', name='fcn3')(drop2)

    # Deconvolutional layers up sampled by 2
    deconv1 = Conv2DTranspose(60, (4, 4), padding='valid', strides=(2, 2), name='deconv1')(fcn3)
    deconv1_cropping = Cropping2D(cropping=((1, 1), (1, 1)), name='deconv1_cropping')(deconv1)
 
    # Fuse a shallow pooling layer
    conv_block4_pool = Conv2D(60, (1, 1), padding='valid', name='conv_block4_pool')(block4_pool)
    added = Add()([deconv1_cropping, conv_block4_pool])
    deconv2 = Conv2DTranspose(60, (32, 32), padding='valid', strides=(16, 16), name='deconv2')(added)
    
    # output pixel-wise probabilities
    deconv2_cropping = Cropping2D(cropping=((8, 8), (8, 8)))(deconv2)
    logits = Conv2D(classes, (1, 1), padding='valid', name='logits')(deconv2_cropping)
    output_pred = Conv2D(1, (1, 1), activation='sigmoid', name='prediction')(logits)
    
        
    return Model(inputs=inputs, outputs=output_pred)