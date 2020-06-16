

import argparse
import xml.dom.minidom
import pathlib
import numpy as np
import openslide
import os, sys
path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
import scipy.misc
from glob import glob
import matplotlib.image as mpimg
from os.path import join, split
from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import xlrd
import openpyxl
from CNN import model_train
import cv2 as cv
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import random
from Get_File_Path import get_file_path
from Load_XML import load_xml
from Get_Ret_Ifo import get_ret_ifo

if not path_wd == '':
    os.chdir(path_wd)
need_save = False
def prepare_data(images_dir_root, size_square):
    num_name = 0
    img_data = []
    image_sum = 0
    img_label = []
    pos = 0


    image_dir_list = glob(join(images_dir_root, r'*/'))
    for image_dir in image_dir_list :
        print("image_dir:",image_dir)
        #image_address = glob(join(image_dir, '*.svs'))
        xml_files = glob(join(image_dir, '*.xml'))
        for index_xml in range(len(xml_files)):
            num_name += 1
            print("xml_files:",xml_files[index_xml])
            xy_list = load_xml(xml_files[index_xml])
            #image_address = get_image_address_from_xml( xml_files[index_xml])
            image_address_list = glob(join(image_dir, '*.ndpi'))
            for image_address_single in image_address_list:
                image_address_single_split_1 = image_address_single.split(".")[:-1]
                print("single_split_:",image_address_single_split_1[0],"   image_address_single: ",image_address_single)
                if image_address_single_split_1[0] in xml_files[index_xml]:
                    image_address = image_address_single
                    break 
            print("image_address:",image_address)
            slide = openslide.open_slide(image_address)
            image_large = get_ret_ifo(xy_list,slide,image_address,size_square,size_square,3,0.2)

            for i in range(len(image_large)):
                image_small = image_large[i]
                for j in range(len(image_small)):
                    img_data.append(image_small[j])
                    #image_sum += 1
                    pos +=1
                    #save image
                    #save_image_address = image_address[0] + str(image_sum) + ".jpg"
                    #cv2.imencode('.jpg', image_small[j])[1].tofile(save_image_address)
                    #image_small[j]. (save_image_address)

            print(num_name)

if __name__ == '__main__':
    print("###########################################this is beginning: \n")
    parser = argparse.ArgumentParser(description='manual to this script', epilog="authors of this script are PengChao YeZixuan XiaoYupei and Ben ")
    #parser.add_argument('--gpus', type=str, default = None)
    #parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--images_dir_root', type=str, default = "/data2/ben/HE/data/TCGA/lung/")
    #parser.add_argument('--labels_address', type=str, default = "../../data/TCGA/lung/labels.csv")
    parser.add_argument('--size_square', type=int, default = 512)
    parser.add_argument('--prepare_types', type=str, default = "1")
    args = parser.parse_args()
    if(args.prepare_types=="1"):
        print("args.images_dir:",args.prepare_types)
        prepare_data(args.images_dir_root, args.size_square)
    elif(args.prepare_types=="2"):
        print("args.images_dir:",args.prepare_types)
        #label_list, image_data_list = select_by_logic.select_by_logic(args.images_dir, args.labels_address, args.need_save_WGI, args.size_square)
 
          
