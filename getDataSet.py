# -*- coding: utf-8 -*-

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
def getDataSet(img_rows,img_cols):
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,9):
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
            img = image.load_img(path+str(i)+"/"+imgList[j], target_size=(img_rows,img_cols))
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
