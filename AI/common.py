#!~/.conda/envs/python36/bin/python
#coding=utf-8
import argparse
import numpy as np
#import tensorflow as tf
import os, sys
path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
from numpy import *
import scipy.misc
#import imageio
from glob import glob
#import matplotlib.image as mpimg
#from PIL import Image
from os.path import join, split
import openslide
#from openslide.deepzoom import DeepZoomGenerator
#from tensorflow_CNN import model_train
import random
import pandas as pd
import cv2
import tensorflow as  tf
#import draw_heatmap
import random

#############
def get_label_list(image_name_list, label_address, label_list,label_name='label'):
        labels_table = pd.DataFrame(pd.read_csv(label_address,names=['file_name', 'label', 'type'], header=0))
        for image_name in image_name_list:
                #print("image_name:",image_name)
                patient_label_info = labels_table.loc[(labels_table['file_name'] == image_name), ['label']]
                if(len(patient_label_info['label'].values) == 0):
                        label_list.append(-1)
                #print("patient_label_info.shape:",patient_label_info.shape)
                #print("patient_label_info:", patient_label_info)
                label_list.append(patient_label_info['label'].values[0])
def get_label(image_name, label_address,header_name='label',ID_prefix_num=15):
        header_name_split = header_name.split(",")
        if(len(header_name_split) == 2):
            header_name_split.append('')
       
        if(len(header_name_split) <= 1):
            return -1

        print("header_name_split---0----",header_name_split[0],"----1----",header_name_split[1],"---2---",header_name_split[2],"----")
        #labels_table = pd.DataFrame(pd.read_csv(label_address,names=[header_name_split[0], header_name_split[1]], header=0))
        print("label_address:",label_address)
        labels_table = pd.DataFrame(pd.read_csv(label_address, header=0))
        #print(labels_table)
        print("=============image_name[0:15]:",image_name[0:15]) #df_sel2=df[df['day'].isin(['fri','mon'])]
        #patient_label_info = labels_table.loc[((labels_table[header_name_split[0]].str.contains(image_name[0:15]))& (labels_table[header_name_split[2]] != 'bak')), [header_name_split[1]]]
        patient_label_info = labels_table.loc[((labels_table[header_name_split[0]].str.contains(image_name[0:ID_prefix_num]))), [header_name_split[1]]]
        if(len(patient_label_info[header_name_split[1]].values) == 0):
                return -1
        else:
                #print("patient_label_info.shape:",patient_label_info.shape)
                #print("patient_label_info:", patient_label_info[header_name])
                return int(patient_label_info[header_name_split[1]].values[0])

def get_address_label_list(image_dir_root, label_address, header_name, image_num, ID_prefix_num, scan_window_suffix='*.png'):
        print("header_name:",header_name)
        image_address_list = []
        image_name_list = []
        label_types = 2
        image_sum = 0
        image_data_list = []
        label_list = []
        image_dir_list = glob(join(image_dir_root, r'*/'))
        #print("svs_dirs:",image_dir_list)
        for image_dir in image_dir_list :
                #print("image_dir:",image_dir)
                image_num1 = image_num
                image_address = glob(join(image_dir, scan_window_suffix))
                len_addr = len(image_address) - 1
                if(len(image_address) == 0):
                        #print("#####################image_address:", image_address)
                        continue
                if len_addr < image_num1:
                        image_num1 = len_addr
                src_list = [random.randint(0, len_addr) for i in range(image_num1)]
                print(src_list)
                for i in range(image_num1):
                        if (len(image_address) < i + 1):
                                continue
                        img_num = src_list[i]
                        image_address_split = image_address[img_num].split("/")
                        print(image_address[img_num])
                        image_name = image_address_split[-1]
                        image_name_split = image_name.split("_")
                        print(image_name_split)
                        svs_name = image_name_split[0]
                        patient_label = -1
                        #patient_label = random.randint(0,1)
                        print("label_address:",label_address)
                        print("svs_name",svs_name)
                        patient_label = get_label(svs_name,label_address,header_name,ID_prefix_num)
                        print("svs_name:",svs_name,"______________patient_label: ",patient_label)
                        if (patient_label == -1):
                                continue
                        print("**************patient_label:",patient_label)
                        ##
                        #if need_save_WGI:
                        #       #https://pypi.org/project/pyvips/2.0.4/
                        #       #img = pyvips.Image.new_from_file(image_address[0], access='sequential')
                        #       #WGI_address = image_address[0] + ".tiff"
                        #       #img.write_to_file(WGI_address)

                        svs_file_slide = openslide.open_slide(image_address[i])
                        image_sum += 1
                        print("image_address[",str(i),"]:",image_address[i])
                        #print("svs_file_slide:", svs_file_slide.shape)

                        #####choose region
                        #svs_address = image_address[i]
                        #region_list = choose_region(svs_file_slide, size_square, svs_address)
                        #image_data = np.array(svs_file_slide.read_region((0, 0), 0, (size_square,size_square)).convert('RGB'))
                        #image_data = np.array(svs_file_slide.read_region((0, 0), 0, (size_square,size_square)))
                        #####choose region

                        image_address_list.append(image_address)
                        image_name_list.append(image_name)
                        label_list.append(patient_label)
                        #image_data_list.append(image_data)

                #for svs_file in  glob(join(svs_dir, '*.svs')):
                #    print(svs_file)
        #get_label_list(image_name_list, label_address,label_list)
        #print("svs_file_list:", image_name_list)
        print(len(image_address_list))
        return label_list , image_address_list

def prepare_features_and_labels(x,y):
    x = tf.cast(x,tf.float32)/255.0
    y = tf.cast(y,tf.int64)
    return x,y

if __name__ == '__main__':
    print("this is a function package")
