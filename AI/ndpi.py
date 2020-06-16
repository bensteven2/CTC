#!/data2/ben/anaconda3/envs/python35/bin/python
#coding=utf-8
import tensorflow as tf
import os, sys
path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
import numpy as np
from numpy import *
import scipy.misc
import imageio
from glob import glob
from PIL import Image
from os.path import join, split
import openslide
from openslide.deepzoom import DeepZoomGenerator
#from tensorflow_CNN import model_train
import random
#import pandas as pd
#import cv2
#from tensorflow.keras import layers
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras

#import tensorflow_CNN
#print(tf.VERSION)
#print(tf.keras.__version__)
import argparse



'''
origin_slide = openslide.open_slide("../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs_43000_49000_10000_5000.ground.png")

origin_data = np.array(origin_slide.read_region((0, 0), 0, (origin_slide.dimensions[0],origin_slide.dimensions[1])))


heatmap_slide = openslide.open_slide("../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs_43000_49000_10000_5000_step_x_50_step_y_50.heatmap.png")

heatmap_data = np.array(heatmap_slide.read_region((0, 0), 0, (heatmap_slide.dimensions[0],heatmap_slide.dimensions[1])))
heatmap_data = heatmap_data.transpose(1,0,2)
#heatmap_data = np.reshape(heatmap_data,(5000,10000,4))
all_data    =   origin_data * 0.4 + heatmap_data * 0.6

imageio.imsave("../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs_43000_49000_10000_5000_step_x_50_step_y_50.compound_test2.png", all_data)




/data2/ben/HE/data/cesc/HE_image/S698161
'''


origin_slide = openslide.open_slide("/data2/ben/HE/data/cesc/HE_image/S698161/S698161-A1.ndpi")

#origin_data = np.array(origin_slide.read_region((0, 0), 0, (origin_slide.dimensions[0],origin_slide.dimensions[1])))
origin_data = np.array(origin_slide.read_region((0, 0), 0, (100,100)))

imageio.imsave("/data2/ben/HE/data/cesc/HE_image/S698161/S698161-A1.png", origin_data)




