from __future__ import absolute_import, division, print_function, unicode_literals
#from skimage import io, transform
import glob
import os
import numpy as np
import time
import xlrd
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter
import cv2
# import xlwt
import os
import tensorflow as tf

from keras import losses
# from tensorflow.keras import layers
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_predict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
###
import tensorflow as tf
#import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras import regularizers
###
'''
def Xception()
tf.keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
'''

def CNN_3(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    model = keras.models.Sequential()
    print("level 1 convolution number(L1):",L1,"convolution size:",F1)
    print("level N pooling size:",F2)
    print("level N convolution number(L2):",L2,"convolution size:",F3)
    model.add(tf.keras.layers.Conv2D(L1, (F1, F1), activation='relu', input_shape=(width,height,channel)))
    for i in range(times):
        model.add(keras.layers.MaxPooling2D((F2, F2)))
        model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)) ##benben
        model.add(keras.layers.Conv2D(L2, (F3, F3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=classes, activation='softmax',kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.01)))
    print(model.summary())
    return model


def VGG_16(width,height,channel,classes,times,L1,L2,F1,F2,F3):

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(width,height,channel), padding='same',
                                  activation='relu', name='conv1_block'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_block'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_block'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_block'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5_block'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_block'))
    model.add(keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv7_block'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_block'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv9_block'))
    model.add(keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='conv10_block'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv11_block'))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv12_block'))
    model.add(keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='conv13_block'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(classes, activation='softmax'))

    return model

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = keras.layers.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Pool(input, pool_size=(3, 3), stride=(1, 1), pool_type='avg',padding='valid',name=None):
    if str.lower(pool_type) == "avg":
        x = keras.layers.AveragePooling2D(pool_size, stride,padding=padding, name=name)(input)
    elif str.lower(pool_type) == 'max':
        x = keras.layers.MaxPooling2D(pool_size, stride,padding=padding, name=name)(input)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = keras.layers.add([x, shortcut])
        return x
    else:
        x = keras.layers.add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = keras.layers.add([x, shortcut])
        return x
    else:
        x = keras.layers.add([x, inpt])
        return x

def resnet34(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    print("width:",width,"height:",height,"channel:",channel,"classes:",classes)
    inpt = keras.layers.Input(shape=(width, height, channel))
    x = keras.layers.ZeroPadding2D((3, 3))(inpt)

    #conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))

    #conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))

    #conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))

    #conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
    x = keras.layers.Flatten()(x)
    activation_function ='tanh' ## 'software'  'relu'
    print("the activation function is:",activation_function)
    #x = keras.layers.Dense(classes, activation='softmax')(x)
    x = keras.layers.Dense(classes, activation=activation_function)(x)

    model = keras.Model(inputs=inpt, outputs=x)
    return model

def resnet50(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    inpt = keras.layers.Input(shape=(width, height, channel))
    x = keras.layers.ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.Model(inputs=inpt, outputs=x)
    return model

def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (5, 5), padding='same', strides=(1, 1), name=None)

    branchpool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

    return x

def GoogleNet(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    inpt = keras.layers.Input(shape=(width, height, channel))
    # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)  # 256
    x = Inception(x, 128)  # 480
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 192)  # 512
    x = Inception(x, 160)
    x = Inception(x, 128)
    x = Inception(x, 112)  # 528
    x = Inception(x, 256)  # 832
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 256)
    x = Inception(x, 384)  # 1024
    x = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.Model(inputs=inpt, outputs=x)
    return model


def Inception_V3(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    base_input = keras.layers.Input(shape=(width, height, channel))  # 299*299*3
    x = Conv2d_BN(base_input, 32, (3, 3), strides=(2, 2), padding='valid', name=None)  # 149*149*32
    x = Conv2d_BN(x, 32, (3, 3), strides=(1, 1), padding='valid', name=None)  # 147*147*32
    x = Conv2d_BN(x, 64,(3, 3), strides=(1, 1), padding='same', name=None)  # 147*147*64
    x = Pool(x, pool_type="max", stride=(2, 2))  # 73*73*64
    x = Conv2d_BN(x, 80,(1, 1), strides=(1, 1), padding='valid', name=None)
    x = Conv2d_BN(x, 192,(3, 3), strides=(1, 1), padding='valid', name=None) # 71*71*192
    x = Pool(x, pool_type="max", stride=(2, 2))  # 35*35*192
    with tf.name_scope("block1") as block1:
        with tf.name_scope("module1") as module1:
            branch1 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64

            branch2 = Conv2d_BN(x, 48,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*48
            branch2 = Conv2d_BN(branch2, 64,(5, 5), strides=(1, 1), padding='same', name=None)  # 35*35*64

            branch3 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64  # 其实可以借用branch1，但这里说明清楚就多写一个
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1,1), padding='same',name = None)  # 35*35*96

            branch4 = Pool(x, pool_type='avg', padding='same')  # 35*35*192
            branch4 = Conv2d_BN(branch4, 32,(1, 1), strides=(1, 1), padding='valid', name=None) # 35*35*32
            x =keras.layers.concatenate(
                [branch1, branch2, branch3, branch4],
                axis=3,
            )  # 35*35*256
            print(x)
        with tf.name_scope("module2") as module2:
            branch1 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64

            branch2 = Conv2d_BN(x, 48,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*48
            branch2 = Conv2d_BN(branch2, 64,(5, 5), strides=(1, 1), padding='same', name=None) # 35*35*64

            branch3 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64 其实可以借用blockModule1_1，但这里说明清楚就多写一个
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96

            branch4 = Pool(x, pool_type='avg', padding='same')  # 35*35*256
            branch4 = Conv2d_BN(branch4, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64
            x = keras.layers.concatenate(
                [branch1, branch2, branch3, branch4],
                axis=3,
            )  # 35*35*288
        with tf.name_scope("module3") as module3:
            branch1 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64

            branch2 = Conv2d_BN(x, 48,(1, 1), strides=(1, 1), padding='valid', name=None) # 35*35*48
            branch2 = Conv2d_BN(branch2, 64,(5, 5), strides=(1, 1), padding='same', name=None)  # 35*35*64

            branch3 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None) # 35*35*64 其实可以借用blockModule1_1，但这里说明清楚就多写一个
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96
            branch3 = Conv2d_BN(branch3, 96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96
            branch4 = Pool(x, pool_type='avg', padding='same')  # 35*35*288
            branch4 = Conv2d_BN(branch4, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64
            x =keras.layers.concatenate(
                [branch1, branch2, branch3, branch4],
                axis=3,
            )  # 35*35*288
    with tf.name_scope("block2") as block2:
        with tf.name_scope("module1") as module1:
            branch1 = Conv2d_BN(x, 384,(3, 3), strides=(2, 2), padding='valid', name=None)  # 17*17*384

            branch2 = Conv2d_BN(x, 64,(1, 1), strides=(1, 1), padding='valid', name=None)  # 35*35*64
            branch2 = Conv2d_BN(branch2,96,(3, 3), strides=(1, 1), padding='same', name=None)  # 35*35*96
            branch2 = Conv2d_BN(branch2,96,(3, 3), strides=(2, 2), padding='valid', name=None) # 17*17*96

            branch3 = Pool(x, pool_type='max', stride=(2, 2))  # 17*17*288
            x = keras.layers.concatenate(
                [branch1, branch2, branch3],
                axis=3,
            )  # 17*17*768
        with tf.name_scope('module2') as module2:
            branch1 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192

            branch2 = Conv2d_BN(x, 128,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*128
            branch2 = Conv2d_BN(branch2, 128,(1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*128
            branch2 = Conv2d_BN(branch2, 192,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192

            branch3 = Conv2d_BN(x, 128,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*768
            # pad:kernel_size // 2
            branch3 = Conv2d_BN(branch3, 128,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*128
            branch3 = Conv2d_BN(branch3, 128,(1, 7), strides=(1, 1), padding='same', name=None) # 17*17*128
            branch3 = Conv2d_BN(branch3, 128,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*128
            branch3 = Conv2d_BN(branch3, 192,(1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*192

            branch4 = Pool(x, pool_type='avg', padding='same')  # 17*17*768
            branch4 = Conv2d_BN(branch4, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            x = keras.layers.concatenate(
                [branch1, branch2, branch3, branch4],
                axis=3,
            )  # 17*17*768
        for i in [3, 4]:
            with tf.name_scope('module'+str(i)):
                branch1 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192

                branch2 = Conv2d_BN(x, 160,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*160
                branch2 = Conv2d_BN(branch2, 160,(1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*160
                branch2 = Conv2d_BN(branch2, 192,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192

                branch3 = Conv2d_BN(x, 160,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*160
                branch3 = Conv2d_BN(branch3, 160,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*160
                branch3 = Conv2d_BN(branch3, 160,(1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*160
                branch3 = Conv2d_BN(branch3, 160,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*160
                branch3 = Conv2d_BN(branch3, 192,(1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*192
                branch4 = Pool(x, pool_type='avg', padding='same')  # 17*17*768
                branch4 = Conv2d_BN(branch4, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
                x = keras.layers.concatenate(
                    [branch1, branch2, branch3, branch4],
                    axis=3,
                )  # 17*17*768
        with tf.name_scope("module5") as module5:
            branch1 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            branch2 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            branch2 = Conv2d_BN(branch2, 192,(1, 7), strides=(1, 1), padding='same',name=None)
            branch2 = Conv2d_BN(branch2, 192,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192

            branch3 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            branch3 = Conv2d_BN(branch3,192,(7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192
            branch3 = Conv2d_BN(branch3, 192, (1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*192
            branch3 = Conv2d_BN(branch3, 192, (7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192
            branch3 = Conv2d_BN(branch3, 192, (1, 7), strides=(1, 1), padding='same', name=None)  # 17*17*192
            branch4 = Pool(x, pool_type='avg', padding='same')  # 17*17*192
            branch4 = Conv2d_BN(branch4, 192, (1, 1), strides=(1, 1), padding='valid', name=None)
            x = keras.layers.concatenate(
                [branch1, branch2, branch3, branch4],
                axis=3,
            )  # 17*17*768
    with tf.name_scope("block3") as block3:
        with tf.name_scope('module1'):
            branch1 = Conv2d_BN(x, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            branch1 = Conv2d_BN(branch1, 320,(3, 3), strides=(2, 2), padding='valid', name=None)  # 8*8*320

            branch2 = Conv2d_BN(x, 192, (1, 1), strides=(1, 1), padding='valid', name=None)  # 17*17*192
            branch2 = Conv2d_BN(branch2, 192, (1, 7), strides=(1, 1), padding='same', name=None)
            branch2 = Conv2d_BN(branch2, 192, (7, 1), strides=(1, 1), padding='same', name=None)  # 17*17*192
            branch2 = Conv2d_BN(branch2, 192,(3, 3), strides=(2, 2), padding='valid', name=None)
            branch3 = Pool(x, pool_type='max', stride=(2, 2))  # 8*8*768
            x = keras.layers.concatenate(
                [branch1, branch2, branch3], axis=3,
            )  # 8*8*1280
        for i in [2, 3]:
            with tf.name_scope('module'+str(i)):
                branch1 = Conv2d_BN(x, 320,(1, 1), strides=(1, 1), padding='valid', name=None)  # 8*8*320

                branch2 = Conv2d_BN(x, 384,(1, 1), strides=(1, 1), padding='valid', name=None) # 8*8*384
                branch2_1 = Conv2d_BN(branch2, 384,(1, 3), strides=(1, 1), padding='same', name=None)# 8*8*384
                branch2_2 = Conv2d_BN(branch2, 384,(3, 1), strides=(1, 1), padding='same', name=None)  # 8*8*384
                branch2 = keras.layers.concatenate([branch2_1, branch2_2], axis=3)  # 8*8*768

                branch3 = Conv2d_BN(x, 448,(1, 1), strides=(1, 1), padding='valid', name=None)  # 8*8*448
                branch3 = Conv2d_BN(branch3, 384,(3, 3), strides=(1, 1), padding='same', name=None)  # 8*8*384
                branch3_1 = Conv2d_BN(branch3, 384,(1, 3), strides=(1, 1), padding='same', name=None)  # 8*8*384
                branch3_2 = Conv2d_BN(branch3, 384,(3, 1), strides=(1, 1), padding='same', name=None)   # 8*8*384
                branch3 = keras.layers.concatenate([branch3_1, branch3_2], axis=3)  # 8*8*768

                branch4 = Pool(x, pool_type='avg', padding='same')  # 6*6*1280
                branch4 = Conv2d_BN(branch4, 192,(1, 1), strides=(1, 1), padding='valid', name=None)  # 8*8*192

                x = keras.layers.concatenate(
                    [branch1, branch2, branch3, branch4],
                    axis=3,
                )  # 8*8*2048

    with tf.name_scope('output'):
        x = keras.layers.GlobalAveragePooling2D()(x)  # 2048
        pred = keras.layers.Dense(classes, activation='softmax')(x)
        model = keras.Model(inputs=base_input, outputs=pred)

    return model


def conv_block(x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
    x = keras.layers.Conv2D(filters=nb_filters,
               kernel_size=(nb_row, nb_col),
               strides=strides,
               padding=padding,
               use_bias=use_bias)(x)
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def stem(x_input):
    x = conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

    x = keras.layers.concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 96, 3, 3, padding='valid')

    x = keras.layers.concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
    x2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    merged_vector = keras.layers.concatenate([x1, x2], axis=-1)
    return merged_vector


def inception_A(x_input):
    """35*35 卷积块"""
    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
        x_input)  # 35 * 35 * 192
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 96, 1, 1)  # 35 * 35 * 96

    conv1x1 = conv_block(x_input, 96, 1, 1)  # 35 * 35 * 96

    conv1x1_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
    conv1x1_3x3 = conv_block(conv1x1_3x3, 96, 3, 3)  # 35 * 35 * 96

    conv3x3_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
    conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
    conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96

    merged_vector = keras.layers.concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3],
                                axis=-1)  # 35 * 35 * 384
    return merged_vector


def inception_B(x_input):
    """17*17 卷积块"""

    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 128, 1, 1)

    conv1x1 = conv_block(x_input, 384, 1, 1)

    conv1x7_1x7 = conv_block(x_input, 192, 1, 1)
    conv1x7_1x7 = conv_block(conv1x7_1x7, 224, 1, 7)
    conv1x7_1x7 = conv_block(conv1x7_1x7, 256, 1, 7)

    conv2_1x7_7x1 = conv_block(x_input, 192, 1, 1)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 192, 1, 7)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 7, 1)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 1, 7)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 256, 7, 1)

    merged_vector = keras.layers.concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
    return merged_vector


def inception_C(x_input):
    """8*8 卷积块"""
    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 256, 1, 1)

    conv1x1 = conv_block(x_input, 256, 1, 1)

    # 用 1x3 和 3x1 替代 3x3
    conv3x3_1x1 = conv_block(x_input, 384, 1, 1)
    conv3x3_1 = conv_block(conv3x3_1x1, 256, 1, 3)
    conv3x3_2 = conv_block(conv3x3_1x1, 256, 3, 1)

    conv2_3x3_1x1 = conv_block(x_input, 384, 1, 1)
    conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 448, 1, 3)
    conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 512, 3, 1)
    conv2_3x3_1x1_1 = conv_block(conv2_3x3_1x1, 256, 3, 1)
    conv2_3x3_1x1_2 = conv_block(conv2_3x3_1x1, 256, 1, 3)

    merged_vector = keras.layers.concatenate(
        [averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
    return merged_vector


def reduction_A(x_input, k=192, l=224, m=256, n=384):
    """Architecture of a 35 * 35 to 17 * 17 Reduction_A block."""
    maxpool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    conv3x3 = conv_block(x_input, n, 3, 3, strides=(2, 2), padding='valid')

    conv2_3x3 = conv_block(x_input, k, 1, 1)
    conv2_3x3 = conv_block(conv2_3x3, l, 3, 3)
    conv2_3x3 = conv_block(conv2_3x3, m, 3, 3, strides=(2, 2), padding='valid')

    merged_vector = keras.layers.concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
    return merged_vector


def reduction_B(x_input):
    """Architecture of a 17 * 17 to 8 * 8 Reduction_B block."""

    maxpool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    conv3x3 = conv_block(x_input, 192, 1, 1)
    conv3x3 = conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')

    conv1x7_7x1_3x3 = conv_block(x_input, 256, 1, 1)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 256, 1, 7)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 7, 1)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    merged_vector = keras.layers.concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
    return merged_vector

def Inception_V4(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    x_input = keras.layers.Input(shape=(width,height,channel))
    # Stem
    x = stem(x_input)  # 35 x 35 x 384
    # 4 x Inception_A
    for i in range(4):
        x = inception_A(x)  # 35 x 35 x 384
    # Reduction_A
    x = reduction_A(x, k=192, l=224, m=256, n=384)  # 17 x 17 x 1024
    # 7 x Inception_B
    for i in range(7):
        x = inception_B(x)  # 17 x 17 x1024
    # Reduction_B
    x = reduction_B(x)  # 8 x 8 x 1536
    # Average Pooling
    x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)  # 1536
    # dropout
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)  # 1536
    # 全连接层
    x = keras.layers.Dense(units=classes, activation='softmax')(x)
    model = keras.Model(inputs=x_input, outputs=x, name='Inception-V4')
    return model

def resnet_v1_stem(x_input):
    x = keras.layers. Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid')(x_input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = keras.layers.Conv2D(80, (1, 1), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(192, (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inception_resnet_v1_C(x_input, scale_residual=True):
    cr1 = keras.layers.Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)

    cr2 = keras.layers.Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)
    cr2 = keras.layers.Conv2D(192, (1, 3), activation='relu', padding='same')(cr2)
    cr2 = keras.layers.Conv2D(192, (3, 1), activation='relu', padding='same')(cr2)

    merged_vector = keras.layers.concatenate([cr1, cr2], axis=-1)

    cr = keras.layers.Conv2D(1792, (1, 1), activation='relu', padding='same')(merged_vector)

    if scale_residual:
        cr = keras.layers.Lambda(lambda x: 0.1 * x)(cr)
    x = keras.layers.add([x_input, cr])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    return x


def reduction_resnet_B(x_input):

    rb1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    rb2 = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb2 = keras.layers.Conv2D(384, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb2)

    rb3 = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb3 =keras.layers. Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb3)

    rb4 =keras.layers. Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(rb4)
    rb4 = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb4)

    merged_vector = keras.layers.concatenate([rb1, rb2, rb3, rb4], axis=-1)

    x =keras.layers. BatchNormalization(axis=-1)(merged_vector)
    x = keras.layers.Activation('relu')(x)
    return x




def resnet_v2_stem(x_input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)  # 149 * 149 * 32
    x = keras.layers.Conv2D(32, (3, 3), activation="relu")(x)  # 147 * 147 * 32
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 147 * 147 * 64

    x1 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x2 = keras.layers.Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

    x1 = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
    x1 = keras.layers.Conv2D(96, (3, 3), activation="relu")(x1)

    x2 = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
    x2 = keras.layers.Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
    x2 = keras.layers.Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
    x2 = keras.layers.Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

    x1 = keras.layers.Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)

    x2 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_A(x_input,n=256, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
    ar1 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)

    ar2 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
    ar2 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)

    ar3 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
    ar3 = keras.layers.Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
    ar3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)

    merged = keras.layers.concatenate([ar1, ar2, ar3], axis=-1)

    ar = keras.layers.Conv2D(n, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: ar = keras.layers.Lambda(lambda a: a * 0.1)(ar) # 是否缩小

    x = keras.layers.add([x_input, ar])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_B(x_input,n1,n2,n3,n4,n5, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    br1 = keras.layers.Conv2D(n1, (1, 1), activation="relu", padding="same")(x_input)

    br2 = keras.layers.Conv2D(n2, (1, 1), activation="relu", padding="same")(x_input)
    br2 = keras.layers.Conv2D(n3, (1, 7), activation="relu", padding="same")(br2)
    br2 = keras.layers.Conv2D(n4, (7, 1), activation="relu", padding="same")(br2)

    merged = keras.layers.concatenate([br1, br2], axis=-1)

    br = keras.layers.Conv2D(n5, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: br = keras.layers.Lambda(lambda b: b * 0.1)(br)

    x = keras.layers.add([x_input, br])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_v2_C(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    cr1 = keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

    cr2 = keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
    cr2 = keras.layers.Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
    cr2 = keras.layers.Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)

    merged = keras.layers.concatenate([cr1, cr2], axis=-1)

    cr = keras.layers.Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: cr = keras.layers.Lambda(lambda c: c * 0.1)(cr)

    x = keras.layers.add([x_input, cr])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def reduction_resnet_A(x_input, k=192, l=224, m=256, n=384):

    ra1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    ra2 = keras.layers.Conv2D(n, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)

    ra3 = keras.layers.Conv2D(k, (1, 1), activation='relu', padding='same')(x_input)
    ra3 = keras.layers.Conv2D(l, (3, 3), activation='relu', padding='same')(ra3)
    ra3 = keras.layers.Conv2D(m, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)

    merged_vector = keras.layers.concatenate([ra1, ra2, ra3], axis=-1)

    x = keras.layers.BatchNormalization(axis=-1)(merged_vector)
    x = keras.layers.Activation('relu')(x)
    return x


def reduction_resnet_v2_B(x_input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    rbr1 =keras.layers. MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)

    rbr2 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr2 = keras.layers.Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)

    rbr3 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr3 = keras.layers.Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)

    rbr4 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr4 = keras.layers.Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
    rbr4 = keras.layers.Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)

    merged = keras.layers.concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
    rbr = keras.layers.BatchNormalization(axis=-1)(merged)
    rbr = keras.layers.Activation("relu")(rbr)
    return rbr

def Inception_resnet_v1(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    x_input = keras.layers.Input(shape=(width,height,channel))
    x = resnet_v1_stem(x_input)

    # 5 x inception_resnet_v1_A
    for i in range(5):
        x = inception_resnet_A(x,256, scale_residual=False)

    # reduction_resnet_A
    x = reduction_resnet_A(x, k=192, l=192, m=256, n=384)

    # 10 x inception_resnet_v1_B
    for i in range(10):
        x = inception_resnet_B(x,128,128,128,128,896, scale_residual=True)

    # Reduction B
    x = reduction_resnet_B(x)

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v1_C(x, scale_residual=True)

    # Average Pooling
    x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)

    # dropout
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=classes, activation='softmax')(x)

    model = keras.Model(inputs=x_input, outputs=x, name='Inception-Resnet-v1')
    return model

def Inception_resnet_v2(width,height,channel,classes,times,L1,L2,F1,F2,F3,scale=True):

    x_input = keras.layers.Input(shape=(width,height,channel))
    x = resnet_v2_stem(x_input)  # Output: 35 * 35 * 256

    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_A(x, 384,scale_residual=scale)
        # Output: 35 * 35 * 256

    # Reduction A
    x = reduction_resnet_A(x, k=256, l=256, m=384, n=384)  # Output: 17 * 17 * 896

    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_B(x, 192,128,160,192,1152,scale_residual=scale)
        # Output: 17 * 17 * 896

    # Reduction B
    x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v2_C(x, scale_residual=scale)
        # Output: 8 * 8 * 1792

    # Average Pooling
    x = keras.layers.AveragePooling2D((8, 8))(x)  # Output: 1792

    # Dropout
    x = keras.layers.Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
    x = keras.layers.Flatten()(x)  # Output: 1792

    # Output layer
    output = keras.layers.Dense(units=classes, activation="softmax")(x)  # Output: 10000

    model = keras.Model(x_input, output, name="Inception-ResNet-v2")
    return model

def _group_conv(x, filters, kernel, stride, groups):

    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    in_channels = keras.backend.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = keras.layers.Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = keras.layers.Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(keras.layers.Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))

    return keras.layers.Concatenate(axis=channel_axis)(gc_list)


def _channel_shuffle(x, groups):

    if keras.backend.image_data_format() == 'channels_last':
        height, width, in_channels = keras.backend.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, height, width, groups, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = keras.backend.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, groups, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]

    x = keras.layers.Lambda(lambda z: keras.backend.reshape(z, pre_shape))(x)
    x = keras.layers.Lambda(lambda z: keras.backend.permute_dimensions(z, dim))(x)
    x = keras.layers.Lambda(lambda z: keras.backend.reshape(z, later_shape))(x)

    return x


def _shufflenet_unit(inputs, filters, kernel, stride, groups, stage, bottleneck_ratio=0.25):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    in_channels = keras.backend.int_shape(inputs)[channel_axis]
    bottleneck_channels = int(filters * bottleneck_ratio)

    if stage == 2:
        x = keras.layers.Conv2D(filters=bottleneck_channels, kernel_size=kernel, strides=1,
                   padding='same', use_bias=False)(inputs)
    else:
        x = _group_conv(inputs, bottleneck_channels, (1, 1), 1, groups)
    x = keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = keras.layers.ReLU()(x)

    x = _channel_shuffle(x, groups)

    x = keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=stride, depth_multiplier=1,
                        padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis)(x)

    if stride == 2:
        x = _group_conv(x, filters - in_channels, (1, 1), 1, groups)
        x = keras.layers.BatchNormalization(axis=channel_axis)(x)
        avg = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(inputs)
        x = keras.layers.Concatenate(axis=channel_axis)([x, avg])
    else:
        x = _group_conv(x, filters, (1, 1), 1, groups)
        x = keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = keras.layers.add([x, inputs])

    return x


def _stage(x, filters, kernel, groups, repeat, stage):
    x = _shufflenet_unit(x, filters, kernel, 2, groups, stage)

    for i in range(1, repeat):
        x = _shufflenet_unit(x, filters, kernel, 1, groups, stage)

    return x


def ShuffleNet(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    inputs = keras.layers.Input(shape=(width,height,channel))

    x = keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = _stage(x, filters=384, kernel=(3, 3), groups=8, repeat=4, stage=2)
    x = _stage(x, filters=768, kernel=(3, 3), groups=8, repeat=8, stage=3)
    x = _stage(x, filters=1536, kernel=(3, 3), groups=8, repeat=4, stage=4)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(classes)(x)
    predicts = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs, predicts)

    return model
#
# def create_model():
#     model = keras.models.Sequential()
#     model.add(keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(512,512,4)))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
#     print(model.summary())
#
#
#     model.add(keras.layers.Flatten())
#     #orig: 64
#     #tmp model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dense(2, activation='softmax'))
#     print(model.summary()) #
#
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#
#     return model

#def build_model():

def create_model_benben(model_types, image_size1, image_size2, image_size3,label_types,times,L1,L2,F1,F2,F3):
    #covn_base = keras.applications.xception.Xception(weights="imagenet", include_top=False, input_shape=(512, 512, 3),pooling="avg")
    covn_base = keras.applications.inception_v3.InceptionV3(input_shape=(image_size1, image_size2, image_size3), include_top=False,weights=None)
    model = keras.Sequential()
    model.add(covn_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(label_types, activation="sigmoid"))
    #
    # for layer in covn_base.layers:
    #     layer.trainable = False

    # # model.add(keras.layers.GlobalAveragePooling2D())
    # model.add(keras.layers.Dense(256, activation="relu"))
    # model.add(keras.layers.Dense(1, activation="sigmoid"))
    # 冰冻卷积层

    #covn_base.trainable = False
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    #
    # # model_train.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # model.fit(images, label, epochs=1, verbose=2)
    covn_base.trainable = True
    set_trainable = False
    for layer in covn_base.layers:
        if layer.name == 'conv2d_109':
            print(layer.name)
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # for i, layer in enumerate(covn_base.layers):
    #     print(i, layer.name)
    #
    # # 我们选择训练最上面的两个 Inception block
    # # 也就是说锁住前面249层，然后放开之后的层。
    # for layer in model.layers[:126]:
    #     layer.trainable = False
    # for layer in model.layers[126:]:
    #     layer.trainable = True

    # model.compile(
    #     optimizer=keras.optimizers.RMSprop(lr=1e-5),  # 以一个较小的学习率微调
    #     loss='binary_crossentropy',
    #     metrics=['acc']
    # )
    print("Adam")
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),  # 以一个较小的学习率微调
        #loss='binary_crossentropy',
        loss=losses.categorical_crossentropy,
        metrics=['acc']
    )

    print("compile model.....")

    return model



################
def create_model(model_types, image_size1, image_size2, image_size3,label_types,times,L1,L2,F1,F2,F3):
    print ("model_types:", model_types, " L1:",L1)
    model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)

    #model.compile(optimizer='adam',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])
    #model.compile(loss=losses.mean_squared_error, optimizer='sgd',metrics=['accuracy'])
    model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
    return model




def draw_roc_curve_validation(test_labels, y_pred,i):
        
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred[:, 1])
    # interp:插值 把结果添加到tprs列表中
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
    plt.plot(fpr, tpr, lw=1, alpha=0.3 )
    plt.savefig("ROC_test.png")
    plt.show()

def draw_roc_curve(test_labels, y_pred,roc_address,roc_title="ROC curve"):
        
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
    # interp:插值 把结果添加到tprs列表中
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    pdf = PdfPages(roc_address)         #先创建一个pdf文件
    plt.figure
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
    plt.plot(fpr, tpr, lw=1, alpha=0.3 )
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ####
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'AUC = %0.2f $\pm$ %0.2f' % (mean_auc,std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)
    plt.legend(loc="lower right")
    ###
    plt.savefig(roc_address)
    plt.show()
    pdf.savefig()                            #将图片保存在pdf文件中
    plt.close()
    pdf.close()                              #这句必须有，否则程序结束pdf文件无法打开

def train_by_all(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2,roc_address="roc.pdf",roc_title="ROC curve"):
    i = 0
    print("-------------------------in the beginning of train_by_all_trainning_set:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    if True:
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images), np.array(train_images)
        Y_train, Y_test = np.array(train_labels), np.array(train_labels)

        Y_train = to_categorical(Y_train, num_classes=None)
        Y_test = Y_train
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("label_types:",label_types)
        model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
        print("|||||||||||||||before compile:")
        # 编译模型
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        #from keras import losses
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ### optimizer='sgd' optimizer='adam'
        #mean_squared_error mean_absolute_percentage_error mean_absolute_error mean_squared_logarithmic_error
        #squared_hinge hinge categorical_hinge logcosh
        # categorical_crossentropy+adam == very good
        #model.compile(loss=losses.categorical_crossentropy, optimizer='adam')
        #model.compile(loss=losses.mean_squared_error, optimizer='adam')
        model.compile(loss=losses.mean_squared_error, optimizer='sgd',metrics=['accuracy'])
        ##below 
        #model.compile(loss=losses.mean_squared_error,  optimizer='adam',metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        draw_roc_curve(Y_test[:,1], probas_[:, 1],roc_address,roc_title=roc_title)
        print("roc_address: ",roc_address)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    del model  # deletes the existing model
    print("-------------after del model")

def cross_validation_1(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2, roc_address="roc_png"):
    
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in cross_validation_1:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    print("label_types:",label_types)
    model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]

        Y_train = to_categorical(Y_train, num_classes=None)
        Y_test = to_categorical(Y_test, num_classes=None)
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        # 编译模型
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        from keras import losses
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ### optimizer='sgd' optimizer='adam'
        #mean_squared_error mean_absolute_percentage_error mean_absolute_error mean_squared_logarithmic_error
        #squared_hinge hinge categorical_hinge logcosh
        # categorical_crossentropy+adam == very good
        model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
        #model.compile(loss=losses.mean_squared_error, optimizer='adam')
        #model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        ##below 
        #model.compile(loss=losses.mean_squared_error,  optimizer='adam',metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test[:,1], probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        #print("Y_test:",Y_test)
        #print("probas_[:, 0]:",probas_[:, 0])
        #print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(roc_address)
    plt.show()

    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    #del model  # deletes the existing model
    print("-------------after del model")
    return mean_auc,model

def cross_validation_by_model_3(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title):
    
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in cross_validation_1:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    print("label_types:",label_types)
    #model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
    #model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
    model=create_model(model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1, L2=L2, F1=F1, F2=F2, F3=F3)
    #model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1, L2=L2, F1=F1, F2=F2, F3=F3, epochs=epochs, batch_size=2, verbose=1)
    pdf = PdfPages(roc_address)         #先创建一个pdf文件
    plt.figure
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]

        #Y_train = to_categorical(Y_train, num_classes=None)
        #Y_test = to_categorical(Y_test, num_classes=None)
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("|||||||||||||||before fit:")
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test[:,1], probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        #print("Y_test:",Y_test)
        #print("probas_[:, 0]:",probas_[:, 0])
        #print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)
    plt.legend(loc="lower right")
    plt.savefig(roc_address)
    plt.show()
    pdf.savefig()                            #将图片保存在pdf文件中
    plt.close()
    pdf.close()
    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    #del model  # deletes the existing model
    print("-------------after del model")
    return mean_auc,model

def Cross_validation(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size=2, roc_address="roc.png"):
    KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    i = 0
    print("-------------------------before split in Cross_validation:")
    print("----------------save_model_address:",save_model_address)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #print("list(KF.split(train_images))",list(KF.split(train_images)))
    for train_index, test_index in KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        print("label_types:",label_types)
        model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1, L2, F1, F2, F3)
        print("|||||||||||||||before compile:")
        # 编译模型
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # 训练模型
        #tf.config.set_device_count222222222 = 3
        print("|||||||||||||||before fit:")
        #print("-----------------------tf.config.device_count333333:",tf.config.device_count)
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
        #fpr, tpr, thresholds = roc_curve(Y_test, probas_)
        #roc_data = (probas_[:,1] , Y_test)
        print("Y_test:",Y_test)
        print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_model_address + "ROC_validation.png")
    plt.show()

    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    ################
    # step6
    #save_model_address = '../../data/result/my_model.H5'
    
    print("save_model_address:", save_model_address)
    model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print("-------------before del model")
    # step7
    del model  # deletes the existing model
    print("-------------after del model")

    
def Keras_Classifier(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3):
    print("begin of keras_classifier: ")
    model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times,L1=L1,L2=L2,F1=F1,F2=F2,F3=F3, epochs=epochs, batch_size=2, verbose=1)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=5000)
    #scores = cross_val_score(model1, train_images, train_labels, cv=kfold)
    print("before of cross_val_predict:")
    y_pre = cross_val_predict(model, train_images, train_labels, cv=kfold)
    print(y_pre)
    y_scores = y_pre
    print("train_labels",train_labels)
    print("y_scores",y_scores)
    fpr, tpr, thresholds = roc_curve(train_labels, y_scores)
    plt.plot(fpr, tpr)
    plt.savefig("ROC.png")
    plt.show()


def cross_validation_select_parameters_3(n_splits,save_model_address,model_types,train_images,train_labels,test_images, test_labels, image_size1, image_size2, image_size3, label_types, epochs, times, L1, L2, F1, F2, F3, batch_size, roc_address, roc_title):
    print("begin of crosss_validation_select_parameters_3: ")
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    # encode class values as integers
    #encoder = LabelEncoder()
    #Y_train_encoder = encoder.fit_transform(Y_train)
    # convert integers to  variables (one hot encoding)
    Y_train = to_categorical(Y_train, 2)
    print("Y_train:", Y_train)    
    for times in range(0,10):
        L1_new= L1 + 10*times
        L2_new= L2 + 10*times

        print("-------------------------------times:",times,"new L1:", L1_new,"new L2:", L2_new)
        roc_address1 = roc_address + "_" +str(times) + ".pdf"
        y_scores_mean,model = cross_validation_by_model_3(n_splits,save_model_address,model_types, X_train, Y_train, X_train, Y_train, image_size1, image_size2, image_size3, label_types, epochs, times, L1_new, L2_new, F1, F2, F3, batch_size, roc_address1, roc_title)
        if y_scores_mean > y_scores_max: 
            times_best = times
            y_scores_max = y_scores_mean
        print("$$$$$$$$$$$$$$$$$$$$$$y_scores_mean:", y_scores_mean, "  y_scores_max:", y_scores_max, "  times:",times, "  times_best:", times_best)
    ####
    L1_best=L1 + 10 * times 
    L2_best=L2 + 10 * times  
    print(" L1_best:", L1_best, " L2_best:", L2_best)
    #print("Y_train",Y_train)    
    #model = eval(model_types)(image_size1, image_size2, image_size3, label_types, times, L1_best, L2_best, F1, F2, F3)
    #model.compile(loss=losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.001, patience=5, verbose=2)
    model=create_model(model_types=model_types, image_size1=image_size1, image_size2=image_size2,label_types=label_types, image_size3=image_size3,times=times, L1=L1_best, L2=L2_best, F1=F1, F2=F2, F3=F3)
    #model = KerasClassifier(build_fn=create_model,model_types=model_types, image_size1=image_size1, image_size2=image_size2, image_size3=image_size3, label_types=label_types, times=times, L1=L1_best, L2=L2_best, F1=F1, F2=F2, F3=F3, epochs=epochs, batch_size=2, verbose=1)
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
    print("before fit ---------Y_train:",Y_train)
    model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_train, Y_train), epochs=epochs,verbose=2,  shuffle=False, callbacks=[early_stopping])
    probas_ = model.predict(X_train)
    print("after prodict-----probas_:",probas_, "label_types: ",label_types)
    fpr, tpr, thresholds = roc_curve(Y_train[:,1], probas_[:, 1])
    
    ####
# Readying neural network model
def build_cnn(activation = 'relu',
              dropout_rate = 0.2,
              optimizer = 'Adam', fs1 = 5, times = 1, init_mode='uniform'):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(10, kernel_size=(fs1, fs1),
              activation=activation,
              input_shape=(120,120,3)))
    for i in range(times):
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(10, (fs1, fs1), activation=activation))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(128, activation=activation))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(units=2, kernel_initializer=init_mode, activation='softmax'))
    print(model.summary())
    model.compile(
        #loss='categorical_crossentropy',
        loss=losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model





