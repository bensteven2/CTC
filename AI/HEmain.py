#!/data2/ben/anaconda3/envs/python35/bin/python
#coding=utf-8
import argparse
import numpy as np
'''
import tensorflow as tf
import os, sys
path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
from numpy import *
import scipy.misc
import imageio
from glob import glob
import matplotlib.image as mpimg
from PIL import Image
from os.path import join, split
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tensorflow_CNN import model_train
import random
import pandas as pd
import cv2
'''
import readHE
import tensorflow_CNN
import draw_heatmap
import CNN

if __name__ == '__main__':

        print("###########################################this is beginning: \n")
        parser = argparse.ArgumentParser(description='manual to this script', epilog="authors of this script are PengChao YeZixuan XiaoYupei and Ben ")
        #parser.add_argument('--gpus', type=str, default = None)
        #parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--images_dir', type=str, default = "../../data/TCGA/lung")
        parser.add_argument('--labels_address', type=str, default = "../../data/TCGA/lung/labels.csv")
        parser.add_argument('--need_save_WGI', type=str, default = "False")
        parser.add_argument('--size_square', type=str, default = 512)
        parser.add_argument('--label_types', type=str, default = 2)
        parser.add_argument('--model_number', type=str, default=0,help="choose the model:0为CNN_3, 1为VGG_16, 2为resnet_34,3为resnet_50, 4为GoogleNet,5为Inception_V3, 6为Inception_V4,7为Inception_resnet_v1, 8为Inception_resnet_v2,9为ShuffleNet")
        parser.add_argument('--epochs', type=str, default=10)
        parser.add_argument('--times', type=str, default=2)
        parser.add_argument('--L1', type=str, default=4,help ="the number of the first conv2d")
        parser.add_argument('--L2', type=str, default=8,help = "the number of the second conv2d ")
        parser.add_argument('--F1', type=str, default=3,help = "the size of the first conv2d layer")
        parser.add_argument('--F2', type=str, default=2,help = "the size of the first maxpooling layer")
        parser.add_argument('--F3', type=str, default=3,help = "the size of the second conv2d layer")


        args = parser.parse_args()
        print("args.labels_address:",args.labels_address)
        print("args.images_dir:",args.images_dir)
        label_list, image_data_list = readHE.HE_process(args.images_dir, args.labels_address, args.need_save_WGI, args.size_square)
        print("#############################################this is the end of read HE! \n")

        model_types = ['CNN_3', 'VGG_16', 'resnet_34', 'resnet_50', 'GoogleNet', 'Inception_V3', 'Inception_V4',
                       'Inception_resnet_v1', 'Inception_resnet_v2','ShuffleNet']

        # model
        #train_images = img_data.reshape((image_sum, image_size1, image_size2, image_size3))
        train_images = np.reshape(image_data_list, (len(image_data_list), args.size_square, args.size_square, 4))

        #test_images = img_data.reshape((image_sum, image_size1, image_size2, image_size3))
        test_images = np.reshape(image_data_list, (len(image_data_list), args.size_square, args.size_square, 4))

        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_labels = label_list
        test_labels = label_list

        print("train_labels:",train_labels)
        CNN.model_train(model_types[args.model_number], train_images, train_labels, test_images, test_labels, args.size_square, args.size_square, 4, args.label_types,args.epochs,args.times,args.L1,args.L2,args.F1,args.F2,args.F3)
        #tensorflow_CNN.model_train(train_images, train_labels, test_images, test_labels, args.size_square, args.size_square, 4, args.label_types)
        #liangyuebin 
        #draw_heatmap.draw_heatmap(args.images_dir, args.labels_address, args.need_save_WGI, args.size_square)



 

