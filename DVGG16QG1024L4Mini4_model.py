#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
as the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
--60000Data-----------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 8, 32, 32)         224
_________________________________________________________________
activation_1 (Activation)    (None, 8, 32, 32)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 32, 32)         0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 32, 32)         584
_________________________________________________________________
activation_2 (Activation)    (None, 8, 32, 32)         0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 16, 16)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 16, 16)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 16, 16)        2336
_________________________________________________________________
activation_3 (Activation)    (None, 32, 16, 16)        0
_________________________________________________________________
dropout_3 (Dropout)          (None, 32, 16, 16)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 16, 16)        9248
_________________________________________________________________
activation_4 (Activation)    (None, 32, 16, 16)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 8, 8)          0
_________________________________________________________________
dropout_4 (Dropout)          (None, 32, 8, 8)          0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1024, 8, 8)        295936
_________________________________________________________________
activation_5 (Activation)    (None, 1024, 8, 8)        0
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024, 8, 8)        0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1024, 8, 8)        9438208
_________________________________________________________________
activation_6 (Activation)    (None, 1024, 8, 8)        0
_________________________________________________________________
dropout_6 (Dropout)          (None, 1024, 8, 8)        0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 1024, 8, 8)        9438208
_________________________________________________________________
activation_7 (Activation)    (None, 1024, 8, 8)        0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1024, 4, 4)        0
_________________________________________________________________
dropout_7 (Dropout)          (None, 1024, 4, 4)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 16384)             0
=================================================================
Total params: 19,184,744
Trainable params: 19,184,744
Non-trainable params: 0
_________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                   
==================================================================================================
input_1 (InputLayer)            (None, 3, 32, 32)    0                                          
__________________________________________________________________________________________________
sequential_1 (Sequential)       (None, 16384)        19184744    input_1[0][0]                  
__________________________________________________________________________________________________
generation (Dense)              (None, 1)            16385       sequential_1[1][0]             
__________________________________________________________________________________________________
auxiliary (Dense)               (None, 10)           163850      sequential_1[1][0]             
==================================================================================================
Total params: 19,364,979
Trainable params: 19,364,979
Non-trainable params: 0
__________________________________________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 2048)              206848
_________________________________________________________________
dense_2 (Dense)              (None, 2048)              4196352
_________________________________________________________________
reshape_1 (Reshape)          (None, 8, 16, 16)         0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 8, 32, 32)         0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 32, 32, 32)        2336
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 3, 32, 32)         387
=================================================================
Total params: 4,405,923
Trainable params: 4,405,923
Non-trainable params: 0
_________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                   
==================================================================================================
input_3 (InputLayer)            (None, 1)            0                                          
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 100)       1000        input_3[0][0]                  
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 100)          0                                          
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 100)          0           embedding_1[0][0]              
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 100)          0           input_2[0][0]                  
                                                                 flatten_2[0][0]                
__________________________________________________________________________________________________
sequential_2 (Sequential)       (None, 3, 32, 32)    4405923     multiply_1[0][0]               
==================================================================================================
Total params: 4,406,923
Trainable params: 4,406,923
Non-trainable params: 0
__________________________________________________________________________________________________
X_train.shape1 (50000, 3, 32, 32)
X_train.shape2 (50000, 3, 32, 32)
component              | loss | generation_loss | auxiliary_loss
-----------------------------------------------------------------
generator (train)      | 1.75 | 1.67            | 0.09
generator (test)       | 1.39 | 1.32            | 0.06
discriminator (train)  | 0.72 | 0.45            | 0.27
discriminator (test)   | 1.00 | 0.45            | 0.56
generated_images (100, 32, 32, 3)

"""
from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
from glob import glob
from keras.layers import MaxPooling2D

np.random.seed(1337)

K.set_image_data_format('channels_first')



def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 32, 32)
    cnn = Sequential()

    cnn.add(Dense(256, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(3 * 8 * 8, activation='relu'))
    cnn.add(Reshape((3, 8, 8)))
    
    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, (3, 3), padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))
    
     # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(32, (3, 3), padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))



    # take a channel axis reduction
    cnn.add(Conv2D(3, (2, 2), padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))
    
    cnn.summary()

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1, ), dtype='int32')

    # 10 classes in MNIST
    num_classes=10
    cls = Flatten()(Embedding(num_classes, latent_size,
                              embeddings_initializer='glorot_normal')(image_class)) #10

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(3, 32, 32)))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(32, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    
    cnn.add(Conv2D(64, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(64, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    
    cnn.add(Conv2D(128,(3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(128, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(128, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    
    cnn.add(Conv2D(256, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(256, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(256, (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())
    cnn.summary()
    
    image = Input(shape=(3, 32, 32))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    num_classes=10
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features) #num_classes=10

    return Model(image, [fake, aux])
