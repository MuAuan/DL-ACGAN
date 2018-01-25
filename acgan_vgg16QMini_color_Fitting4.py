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
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
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
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Adamax
from keras.utils.generic_utils import Progbar
import numpy as np
from keras.datasets import cifar10, cifar100  #10
import keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from DVGG16QG1024L4Mini4_model import build_generator, build_discriminator
#from DQGQ_model import build_generator, build_discriminator

from keras.preprocessing import image
import sys
import cv2
import os


np.random.seed(1337)

K.set_image_data_format('channels_first')

#その１　------データセット作成------

#フォルダは整数で名前が付いています。
def getDataSet():
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,4):
        path = "./train_images"
        if i == 0:
            #othersは15000枚用意します。テスト用には3000枚
            cutNum = 480
            cutNum2 = 440
        else:
            #主要キャラたちは1800枚ずつ。テスト用には300枚
            cutNum = 480
            cutNum2 = 440
        imgList = os.listdir(path+str(i))
        print(imgList)
        imgNum = len(imgList)
        for j in range(cutNum):
            #imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
            img = image.load_img(path+str(i)+"/"+imgList[j], target_size=(64,64))
            imgSrc = image.img_to_array(img)
            #imreadはゴミを吸い込んでも、エラーで止まらずNoneを返してくれます。
            #ですので読み込み結果がNoneでしたらスキップしてもらいます。
            if imgSrc is None:continue
            if j < cutNum2:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)
    print(len(X_train),len(y_train),len(X_test),len(y_test))

    return X_train,y_train,X_test,y_test



if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs = 200
    batch_size = 32 #32
    latent_size = 100
    num_classes = 10 #10
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    #generator parames of Adamax
    adam_lr1 = 0.0000125 #0.0001
    adam_beta_11 = 0.5
    
    #discriminator params of Adam
    adam_lr = 0.00002 #0.0002
    adam_beta_1 = 0.5
    
    #keras default Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    #keras default Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #keras default RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    
    # build the discriminator
    discriminator = build_discriminator()
    #discriminator.load_weights('params_discriminator_akb10_epoch_101.hdf5')
    #discriminator.trainable = False
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=0.999, epsilon=1e-08, decay=0.00001),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()
    
    # build the generator
    generator = build_generator(latent_size)
    #generator.load_weights('params_generator_akb10_epoch_101.hdf5')
    #generator.trainable = False
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=0.999, epsilon=1e-08, decay=0.),
                      loss='binary_crossentropy')
    generator.summary()
    
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine') #10
    
    
    #上で作った関数でデータセットを用意します。
    #x_train,y_train,x_test,y_test = getDataSet()
    #print(x_train)
    #print(np.array(x_train))
    
    #このままだとchainerで読み込んでもらえないので、array型にします。
    #x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3, 32, 32)) / 255
    x_train = np.array(x_train) # / 255
    y_train = np.array(y_train).astype(np.int32)
    x_test = np.array(x_test) #/ 255
    y_test = np.array(y_test).astype(np.int32)
    
   
    X_train=x_train[0:10000]  #.transpose(0,2,3,1) 1760
    print("X_train.shape1",X_train.shape)
    X_test=x_test[0:1000]   #.transpose(0,2,3,1) 160
    y_train=y_train[0:10000].reshape(1,10000)[0] #1750
    y_test=y_test[0:1000].reshape(1,1000)[0]
    
    for i in range(13):
        plt.figure(num=None, figsize=(10, 10), dpi=50)
        plt.imshow(X_train.transpose(0,2,3,1)[i][0:63][0:63])
        print(y_train[i])
        plt.pause(1)
        plt.close()
        #plt.imshow(X_train.transpose(0,2,3,1)[i][0:31][0:31])        
        #plt.pause(1)
        #plt.close()
    
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    print("X_train.shape2",X_train.shape)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))
        if epoch%1:
            discriminator.trainable = True
            discriminator.compile(
                optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=0.999, epsilon=1e-08, decay=0.),
                loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
                )
        else:
            discriminator.trainable = True
            discriminator.compile(
                optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=0.999, epsilon=1e-08, decay=0.),
                loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
                )

        
        num_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size,latent_size))
            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            # sample some labels from p_c
            sampled_labels = np.random.randint(0, num_classes, batch_size) #num_classes=10

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)  #num_classes=10

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (num_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, num_classes, num_test) #num_classes=10
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        
        
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)
        
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)  #num_classes=10

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_cifar10_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_cifar10_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (25, latent_size))

        sampled_labels = np.array([
            [i] * 5 for i in range(5)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)
        print("generated_images",generated_images.transpose(0,2,3,1).shape)
        
        #X, Y = np.meshgrid(range(32),range(32))
        #Z = np.zeros((10,10))
        plt.figure(num=None, figsize=(30, 30), dpi=60) #, facecolor='w', edgecolor='k')
        for j in range(5):
            for i in range(5):
                plt.subplot(5,5,1+i+5*j)
                plt.imshow(generated_images.transpose(0,2,3,1)[i+5*j][0:63][0:63],interpolation='bilinear') #interpolation='bilinear' interpolation='nearest'
                plt.axis('off')
                plt.subplots_adjust(hspace = .001)
                #plt.tight_layout()
        plt.pause(3)
        plt.savefig('plot_epoch_{0:03d}_cifar_generated.png'.format(epoch), dpi=60)
        plt.close()
        """
        sampled_labels = np.array([
            [i] * 2 for i in range(2)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)
        print("generated_images",generated_images.transpose(0,2,3,1).shape)
        
        #X, Y = np.meshgrid(range(32),range(32))
        #Z = np.zeros((10,10))
        plt.figure(num=None, figsize=(90, 90), dpi=60) #, facecolor='w', edgecolor='k')
        for j in range(2):
            for i in range(2):
                plt.subplot(2,2,1+i+2*j)
                plt.imshow(generated_images.transpose(0,2,3,1)[i+2*j][0:31][0:31],interpolation='bilinear') 
                plt.axis('off')
                plt.subplots_adjust(hspace = .001)
        plt.pause(3)
        plt.savefig('plot_epoch_{0:03d}_generated.png'.format(epoch), dpi=60)
        plt.close()
        """




    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))

