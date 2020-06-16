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
import scipy.misc as misc
from glob import glob
from PIL import Image
from os.path import join, split
import openslide
from openslide.deepzoom import DeepZoomGenerator
#from tensorflow_CNN import model_train 
import random
import pandas as pd
import cv2
import common
from color_norm import color
import argparse
from get_area_ratio import get_area_ratio
#from tensorflow.keras import layers
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras

#import tensorflow_CNN 
#print(tf.VERSION)
print(tf.keras.__version__)

#import pyvips
##############   
dir(scipy.misc)
if not path_wd == '':
    os.chdir(path_wd)
    
#print("os.getcwd():",os.getcwd())
#need_save_WGI = True

need_show = False


def get_label_list(image_name_list, label_address,label_list):

	labels_table = pd.DataFrame(pd.read_csv(label_address,names=['file_name', 'label', 'type'], header=0))
	for image_name in image_name_list:
		#print("image_name:",image_name)
		label_name = labels_table.loc[(labels_table['file_name'] == image_name), ['label']]
		if(len(label_name['label'].values) == 0):
			label_list.append(-1)
		#print("label_name.shape:",label_name.shape)	
		#print("label_name:", label_name)
		label_list.append(label_name['label'].values[0])
def get_label(image_name, label_address):
	labels_table = pd.DataFrame(pd.read_csv(label_address,names=['file_name', 'label', 'type'], header=0))
	#print("image_name:",image_name)
	label_name = labels_table.loc[((labels_table['file_name'] == image_name) & (labels_table['type'] != 'bak')), ['label']]
	if(len(label_name['label'].values) == 0):
		return -1
	else:
		#print("label_name.shape:",label_name.shape)	
		#print("label_name:", label_name['label'])
		return int(label_name['label'].values[0])

def save_image(svs_save_file_name, image_data, need_show=False):
	
	#if os.path.exists(svs_save_file_name):
	#	return
	#If you have Pillow installed with scipy and it is still giving you error then check your scipy version 
	#because it has been removed from scipy since 1.3.0rc1.
	#rather install scipy 1.1.0 by :
	#pip install scipy==1.1.0
	#scipy.misc.imsave(svs_save_file_name, image_data)
	imageio.imsave(svs_save_file_name, image_data)
	if(need_show):
		im = Image.fromarray(image_data)
		im.show()

	## read and show a png file:
	#lena = mpimg.imread('lena.png') #
	#im = Image.fromarray(np.uinit8(lena*255))
	#im = Image.fromarray(img)   
	#im.show()

def draw_heatmap_multiple(images_dir_root, label_address, header_name, save_model_address, step_x, step_y, scan_window_suffix, ID_prefix_num):
	label_list, image_address_list = common.get_address_label_list(images_dir_root, label_address, header_name, 10, ID_prefix_num, scan_window_suffix)
	for image_address_index in range(len(image_address_list)):
		print("image_address_index:",image_address_index,"image_address:",image_address_list[image_address_index], " label:", label_list[image_address_index])
		draw_heatmap(image_address_list[image_address_index],save_model_address,size_square,0,0,-1,-1,step_x,step_y)	
		
 
def draw_heatmap(image_address, save_model_address, size_square, begin_x, begin_y, dimensions_x, dimensions_y, step_x, step_y):
	image_size1 = size_square
	image_size2 = size_square
	image_size3 = 4
	label_types = 2
	image_sum = 0
	image_data_list = []
	label_list = [] 
	#print("svs_dirs:",image_dir_list)
	svs_file_slide = openslide.open_slide(image_address)

	read_region_1 = 10000    #left
	read_region_2 = 0    #up
	read_region_3 = read_region_1 + size_square    ##x axis right
	read_region_4 = read_region_2 + size_square    ##y axis down
	region_list = []
	WGI_dim0 = svs_file_slide.dimensions[0]
	WGI_dim1 = svs_file_slide.dimensions[1]
	print(svs_file_slide.dimensions)
	if(dimensions_x == -1):
		dimensions_x = svs_file_slide.dimensions[0]
	if(dimensions_y == -1):
		dimensions_y = svs_file_slide.dimensions[1]	
	#print(svs_file_slide.dimensions[0])
 	#print("svs_file_slide:", svs_file_slide.dimensions)
	#print("svs_file_slide", svs_file_slide)
	#image_data0 = np.array(svs_file_slide.read_region((1000,1000), 0, (size_square, size_square)))
	#top1_list = [0,0,0,image_data0,image_data0]
	#top2_list = [0,0,0,image_data0,image_data0]
	#top3_list = [0,0,0,image_data0,image_data0]

	###  10000  10000  to 20000 20000
	#begin_x = 43000
	#begin_y = 49000
	#dimensions_x = 10000  # svs_file_slide.dimensions[0]
	#dimensions_y = 5000  # svs_file_slide.dimensions[1]
	#step_x = 50
	#step_y = 50

	end_x = begin_x + dimensions_x # 
	end_y =  begin_y + dimensions_y   #
	heatmap_image = np.ones([dimensions_x,dimensions_y,3]) * 255
	heatmap_array = np.zeros([dimensions_x,dimensions_y])
	#ones_array = np.ones([dimensions_x,dimensions_y])
	#heatmap_array_times = np.zeros((svs_file_slide.dimensions[0],svs_file_slide.dimensions[1]))
	heatmap_array_times = np.ones([dimensions_x,dimensions_y])
	#heatmap_array_times = np.zeros([dimensions_x,dimensions_y])
	#restored_model = tf.keras.models.load_model(save_model_address)

	for slide_x in range(0,dimensions_x-size_square-1,step_x):
		for slide_y in range(0,dimensions_y-size_square-1,step_y):
			#image_data = np.array(svs_file_slide.read_region((slide_x + begin_x, slide_y + begin_y), 0, (size_square, size_square)))
			image_data = svs_file_slide.read_region((slide_x + begin_x, slide_y + begin_y), 0,
													(size_square, size_square)).convert('RGB')
			#print("image_data.dimensions:", image_data.shape)
			ratio = get_area_ratio(image_data)
			if ratio > 0.9: ###benben 0.2
				pred = [0]

			else:

				image_data = np.reshape(image_data, (1, size_square, size_square, 3))
				test_image = np.array(image_data / 255.0)

				print("shape of image_data_list:",shape(image_data_list[:1]))
				#test_image = np.reshape(image_data, (1, size_square, size_square, 4))
				restored_model = tf.keras.models.load_model(save_model_address)
				pred = restored_model.predict(test_image)[:,1]
			# print("shape of pred[0]:",shape(pred))
			# print('predict[0,1]:',pred)
			#print("test_image", test_image)
			
			#for prob_x in range(slide_x, slide_x + size_square):
			#	for prob_y in range(slide_y, slide_y + size_square):
			#		heatmap_array[prob_x,prob_y] = heatmap_array[prob_x,prob_y] + pred[0,1]
			#		heatmap_array_times[prob_x,prob_y] = heatmap_array_times[prob_x,prob_y] + 1 
			heatmap_array[slide_x:slide_x + size_square, slide_y:slide_y + size_square] +=  pred
			heatmap_array_times[slide_x:slide_x + size_square, slide_y:slide_y + size_square] += 1 
			
			print("-----slide_x:",str(slide_x) + "   slide_y:",str(slide_y),"   pred:", pred, "   ratio:",ratio)
	print("heatmap_array_times:",heatmap_array_times)			 
	ground_image = svs_file_slide.read_region((begin_x, begin_y), 0, (dimensions_x,dimensions_y)).convert('RGB')            
	heatmap_array = heatmap_array/heatmap_array_times
	heatmap_image[:,:,0] = heatmap_image[:,:,0] * heatmap_array 
	heatmap_image[:,:,1] = heatmap_image[:,:,1] * 0 # (ones_array - heatmap_array)
	#heatmap_image[:,:,2] = heatmap_image[:,:,2] * (ones_array - heatmap_array)
	heatmap_image[:,:,2] = heatmap_image[:,:,2] * (1 - heatmap_array)
	print("heatmap_image:",heatmap_image)
	heatmap_address = image_address + "_" + str(begin_x) + "_" +  str(begin_y) + "_" + str(dimensions_x) + "_" +  str(dimensions_y) + "_step_x_" +  str(step_x) + "_step_y_" + str(step_y)  +".heatmap.png"
	ground_address = image_address + "_" + str(begin_x) + "_" +  str(begin_y) + "_" + str(dimensions_x) + "_" +  str(dimensions_y) + ".ground.png"
	compound_address = image_address + "_" + str(begin_x) + "_" +  str(begin_y) + "_" + str(dimensions_x) + "_" +  str(dimensions_y) + "_step_x_" +  str(step_x) + "_step_y_" + str(step_y)  +".compound.png"
	#plt.imsave(heatmap_address,your_numpy_array,cmap='gray')
	#cv2.imwrite(heatmap_address,heatmap_image)

	heatmap_image = heatmap_image.transpose(1,0,2)
	save_image(heatmap_address, heatmap_image, False)
	ground_image.save(ground_address)

	ground_image = np.array(ground_image)
	heatmap_image = np.array(heatmap_image)
	#heatmap_image = np.reshape(heatmap_image,(dimensions_x,dimensions_y,3))
	compound_image = ground_image * 0.4 + heatmap_image * 0.6
	imageio.imsave(compound_address, compound_image)


if __name__ == '__main__':

        print("###########################################this is beginning: \n")
        parser = argparse.ArgumentParser(description='manual to this script', epilog="authors of this script are PengChao YeZixuan XiaoYupei and Ben ")
        #parser.add_argument('--gpus', type=str, default = None)
        #parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--images_dir_root', type=str, default = "../../data/TCGA/lung")
        parser.add_argument('--labels_address', type=str, default = "reg-tmb.csv")
        parser.add_argument('--header_name', type=str, default = "id,labels")
        parser.add_argument('--scan_window_suffix', type=str, default = "*.ndpi")
        parser.add_argument('--pattern', type=str, default = "single")
        parser.add_argument('--images_address', type=str, default = "../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs")
        #parser.add_argument('--save_model_address', type=str, default = "../../data/result/my_model.CNN_3.bak")
        parser.add_argument('--save_model_address', type=str, default = "../../data/result/my_model.h5")

        parser.add_argument('--size_square', type=str, default = 512)
        parser.add_argument('--label_types', type=str, default = 2)
        parser.add_argument('--model_number', type=str, default=0,help="choose the model:0为CNN_3, 1为VGG_16, 2为resnet_34,3为resnet_50, 4为GoogleNet,5为Inception_V3, 6为Inception_V4,7为Inception_resnet_v1, 8为Inception_resnet_v2,9为ShuffleNet")
        parser.add_argument('--epochs', type=str, default= 10)
        parser.add_argument('--times', type=str, default= 2)
        parser.add_argument('--begin_x', type=int, default= 43000)
        parser.add_argument('--begin_y', type=int, default= 49000)
        parser.add_argument('--dimensions_x', type=int, default= 10000)
        parser.add_argument('--dimensions_y', type=int, default= 5000)
        parser.add_argument('--step_x', type=int, default= 500)
        parser.add_argument('--step_y', type=int, default= 500)
        parser.add_argument('--ID_prefix_num', type=int, default=15)

        args = parser.parse_args()
        print("args.images_dir_root:",args.images_dir_root)
        print("args.step_x:",args.step_x,"args.step_y:",args.step_y)
        if(args.pattern=="single"):
            draw_heatmap(args.images_address, args.save_model_address, args.size_square, args.begin_x, args.begin_y, args.dimensions_x, args.dimensions_y, args.step_x, args.step_y)
        elif(args.pattern=="multiple"):
            draw_heatmap_multiple(args.images_dir_root, args.labels_address, args.header_name, args.save_model_address, args.step_x, args.step_y, args.scan_window_suffix, args.ID_prefix_num) 

