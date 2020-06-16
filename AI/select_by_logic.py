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
import matplotlib.image as mpimg
from PIL import Image
from os.path import join, split
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tensorflow_CNN import model_train 
import random
import pandas as pd
import cv2
#from tensorflow.keras import layers
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras

import tensorflow_CNN 
print(tf.VERSION)
print(tf.keras.__version__)
import argparse
import pyvips
##############   
dir(scipy.misc)
if not path_wd == '':
    os.chdir(path_wd)
    
#print("os.getcwd():",os.getcwd())
#need_save_WGI = True

need_show = False

###########

def get_gray(img):
	#img=cv2.imread(path)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#print("gray:", gray)
	return gray

def Gaussian_Blur(gray):
    #
    blurred = cv2.GaussianBlur(gray, (5, 5),0)

    return blurred

def Sobel_gradient(blurred):
    #
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

def Thresh_and_blur(gradient,threshold1):

    blurred = cv2.GaussianBlur(gradient, (5,5),0)
    #(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    (_, thresh) = cv2.threshold(blurred, threshold1, 255, cv2.THRESH_BINARY_INV)
    return thresh


def image_morphology(thresh):
    # 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    #print(" kernel is",kernel)
    #
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 
    closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) 
    
    closed = cv2.erode(closed, kernel, iterations=2)
    #closed = cv2.dilate(closed, kernel, iterations=1)
    

    return closed

def findcnts_and_box_point(closed, image_size_threshold = 0.0):
    #
    #print("close.copy():", closed.copy())
    (cnts, _) = cv2.findContours(closed.copy(), 
        #cv2.RETR_LIST, 
	cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    #print (" cnts: ",cnts)
    sorted_list = sorted(cnts, key=cv2.contourArea, reverse=True)
    box_list = []
    if(len(sorted_list)>0):
        for list_index in sorted_list:
            c = list_index
            rect = cv2.minAreaRect(c)
            #print("rect[1][0]: ",rect[1][0],"rect[1][1]: ",rect[1][1],"area: ", rect[1][0] * rect[1][1]  )
            if  rect[1][0] * rect[1][1] > image_size_threshold:
                #print (" enter: ", "image_size_threshold = ", image_size_threshold)
                box_list.append(np.int0(cv2.boxPoints(rect)))
            
    #c = sorted(cnts, key=cv2.contourArea, reverse=True)[2]
    # compute the rotated bounding box of the largest contour
    #rect = cv2.minAreaRect(c)
    #box = np.int0(cv2.boxPoints(rect))

    return box_list, cnts

def drawcnts_and_cut(original_img, box_list):
    draw_img = original_img.copy()
    crop_img_list = []
    for box in box_list:
        # 
        # draw a bounding box arounded the detected barcode and display the image
        draw_img = cv2.drawContours(draw_img, [box], -1, (0, 0, 255), 3)
    
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)     
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        x1 -= 2
        x2 += 2
        y1 -= 2
        y2 += 2
        hight = y2 - y1
        width = x2 - x1
        crop_img = original_img[y1:y1+hight, x1:x1+width]
        #print("x1",x1,"x2",x2,"y1",y1,"y2",y2,"hight",hight,"width",width, "area: ", hight * width)
        crop_img_list.append(crop_img)

    return draw_img, crop_img_list

def image_process(img,svs_address,threshold1, image_size_threshold = 0.0):
	
	#original_img, gray = get_image(img)
	img_3color =img[:,:,0:3]
	#print("img.shape: ",img[:,:,0:3].shape)
	original_img = img_3color
	gray = get_gray(img_3color)
	#if original_img.all() ==None:
	#return
	#blurred = Gaussian_Blur(img_3color)
	blurred = Gaussian_Blur(gray)
	
	gradX, gradY, gradient = Sobel_gradient(blurred)
	gradient = blurred
	
	thresh = Thresh_and_blur(gradient,threshold1)
	closed = image_morphology(thresh)
	box_list,cnts = findcnts_and_box_point(closed, image_size_threshold)
	draw_img = img_3color.copy()
	for box in box_list: 
	     draw_img, crop_img_list = drawcnts_and_cut(original_img,box_list)
	
	#
	
	#draw_img1 = img_3color.copy()
	#gray = cv2.cvtColor(draw_img1,cv2.COLOR_BGR2GRAY)  
	#ret, binary = cv2.threshold(gray,threshold1,255,cv2.THRESH_BINARY)  
  
	#contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
	#draw_img2 = img_3color.copy()
	#cv2.drawContours(draw_img2,contours,-1,(255,0,255),3)  
	
	draw_img2 = img_3color.copy()
	#print("$$$$$$$$$$$$ draw_img2.shape:",draw_img2.shape,"contours.shape:",len(cnts))
	cv2.drawContours(draw_img2,cnts,-1,(0,0,255),1)
	
	color_img = img_3color.copy()
	zeros_img_2 = zeros(list(color_img.shape[:2]))
	red_img = color_img.copy()
	red_img[:,:,0] = zeros_img_2
	red_img[:,:,1] = zeros_img_2
	green_img = color_img.copy()
	green_img[:,:,0] = zeros_img_2
	green_img[:,:,2] = zeros_img_2
	blue_img = color_img.copy()
	blue_img[:,:,1] = blue_img[:,:,1]*0
	blue_img[:,:,2] = blue_img[:,:,2]*0
	#print("blue_img",blue_img)
	#print("red_img",red_img)
	#print("max:",color_img.max(),"min:",color_img.min(),"mean:",color_img.mean())

	#print("format:",color_img.format,"mode:",color_img.mode,"type:",type(color_img))
	#if(save_result): ###save  Intermediate process
	#
	#from skimage import io,data
	#img=data.chelsea()
	#reddish = img[:, :, 0] >170
	#img[reddish] = [0, 255, 0]
	#io.imshow(img)
	#
	'''
	from skimage import exposure,data
	hist1=np.histogram(color_img, bins=2) #用numpy包计算直方图
	hist2=exposure.histogram(color_img, nbins=2) #用skimage计算直方图
	print(hist1)
	#print(hist2)
	#####

	import matplotlib.pyplot as plt
	img=data.camera()
	plt.figure("hist")
	arr=img.flatten()
	n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red') 
	plt.show()
	'''
	#####
	#img=data.lena()
	#ar=img[:,:,0].flatten()
	#plt.hist(ar, bins=256, normed=1,facecolor='r',edgecolor='r',hold=1)
	#ag=img[:,:,1].flatten()
	#plt.hist(ag, bins=256, normed=1, facecolor='g',edgecolor='g',hold=1)
	#ab=img[:,:,2].flatten()
	#plt.hist(ab, bins=256, normed=1, facecolor='b',edgecolor='b')
	#plt.show()



	#####
	'''
	cv2.imencode('.jpg',blurred)[1].tofile(svs_address + "blurred.jpg")	
	cv2.imencode('.jpg',gradX)[1].tofile(svs_address + "gradX.jpg")
	cv2.imencode('.jpg',gradY)[1].tofile(svs_address + "gradY.jpg")
	cv2.imencode('.jpg',gradient)[1].tofile(svs_address + "gradient.jpg")
	cv2.imencode('.jpg',thresh)[1].tofile(svs_address + "thresh.jpg")
	cv2.imencode('.jpg',closed)[1].tofile(svs_address + "closed.jpg")	
	cv2.imencode('.jpg',draw_img)[1].tofile(svs_address + "drawed.jpg")	
	cv2.imencode('.jpg',draw_img2)[1].tofile(svs_address + "drawed2.jpg")	
	cv2.imencode('.jpg',red_img)[1].tofile(svs_address + "red.jpg")	
	cv2.imencode('.jpg',green_img)[1].tofile(svs_address + "green.jpg")	
	cv2.imencode('.jpg',blue_img)[1].tofile(svs_address + "blue.jpg")	
        '''
	
	return draw_img2, cnts
	
	

#############
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

def save_image(svs_save_file_name, image_data, need_show):
	
	if os.path.exists(svs_save_file_name):
		return
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
def choose_region(svs_file_slide, size_square, svs_address,save_model_address):
	read_region_1 = 10000    #left
	read_region_2 = 0    #up
	read_region_3 = read_region_1 + size_square    ##x axis right
	read_region_4 = read_region_2 + size_square    ##y axis down
	region_list = []
	WGI_dim0 = svs_file_slide.dimensions[0]
	WGI_dim1 = svs_file_slide.dimensions[1]
	print(svs_file_slide.dimensions)
	#print(svs_file_slide.dimensions[0])
 	#print("svs_file_slide:", svs_file_slide.dimensions)
	#print("svs_file_slide", svs_file_slide)
	image_data0 = np.array(svs_file_slide.read_region((1,1), 0, (size_square, size_square)))
	top1_list = [0,0,0,image_data0,image_data0]
	top2_list = [0,0,0,image_data0,image_data0]
	top3_list = [0,0,0,image_data0,image_data0]
	for slide_x in range(1,svs_file_slide.dimensions[0],4000):
		for slide_y in range(1, svs_file_slide.dimensions[1], 4000):
			image_data = np.array(svs_file_slide.read_region((slide_x, slide_y), 0, (size_square, size_square)))
			#print("image_data.dimensions:", image_data.shape)

			##
			save_model_address = '../../data/result/my_model.h5'

			restored_model = tf.keras.models.load_model(save_model_address)
			image_data_list = []
			image_data_list.append(image_data)
			print("shape of image_data_list:",shape(image_data_list[:1]))
			test_images = np.reshape(image_data_list, (len(image_data_list), size_square, size_square, 4))
			pred = restored_model.predict(test_images)
			print("shape of pred[0]:",shape(pred[0,1]))
			print('predict:',pred[0,1])

			##
			print(svs_address," x:",slide_x," y:",slide_y)
			svs_address_slide = svs_address + "." +  str(slide_x) + "_" +  str(slide_y) + "."
			threshold1 = 70
			draw_img2, cnts =image_process(image_data, svs_address_slide, threshold1)
			if len(cnts) >= top1_list[0]:
				top3_list = top2_list.copy()
				top2_list = top1_list.copy()
				top1_list[0] = len(cnts)
				top1_list[1] = slide_x
				top1_list[2] = slide_y
				top1_list[3] = image_data 
				top1_list[4] = draw_img2
			elif len(cnts) >= top2_list[0]:
				top3_list = top2_list.copy()
				top2_list[0] = len(cnts)
				top2_list[1] = slide_x
				top2_list[2] = slide_y
				top2_list[3] = image_data 
				top2_list[4] = draw_img2
			elif len(cnts) > top3_list[0]:
				top3_list[0] = len(cnts)
				top3_list[1] = slide_x
				top3_list[2] = slide_y
				top3_list[3] = image_data 
				top3_list[4] = draw_img2
	topn_list = top1_list
	top_label = "top1"
	svs_save_original_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) + top_label + '.original.jpg')
	svs_save_drawed_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) + top_label + '.drawed.jpg')
	cv2.imencode('.jpg', topn_list[3])[1].tofile(svs_save_original_file_name)	
	cv2.imencode('.jpg', topn_list[4])[1].tofile(svs_save_drawed_file_name)	
	
	topn_list = top2_list
	top_label = "top2"
	svs_save_original_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) + top_label +  '.original.jpg')
	svs_save_drawed_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) +  top_label + '.drawed.jpg')
	cv2.imencode('.jpg', topn_list[3])[1].tofile(svs_save_original_file_name)	
	cv2.imencode('.jpg', topn_list[4])[1].tofile(svs_save_drawed_file_name)	

	topn_list = top3_list
	top_label = "top3"
	svs_save_original_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) + top_label + '.original.jpg')
	svs_save_drawed_file_name = (svs_address + "-" +str(WGI_dim0) + "_" + str(WGI_dim1) + "-"   +  str(topn_list[1]) + "_" +  str(topn_list[2]) +  top_label + '.drawed.jpg')
	cv2.imencode('.jpg', topn_list[3])[1].tofile(svs_save_original_file_name)	
	cv2.imencode('.jpg', topn_list[4])[1].tofile(svs_save_drawed_file_name)	
	#if need_save_WGI:
	#	save_image(svs_save_file_name, image_data, need_show)
	#	print("save_image_file_name:", svs_save_file_name)	
	#print("#########################image_data.shape:", image_data.shape)
	region_list.append(top1_list[3])
	return region_list

def HE_process(image_dir_root, label_address, need_save_WGI, size_square,save_model_address):
	image_address_list = []
	image_name_list = []
	image_size1 = size_square
	image_size2 = size_square
	image_size3 = 4
	label_types = 2
	image_sum = 0
	image_data_list = []
	label_list = [] 
	image_dir_list = glob(join(image_dir_root, r'*/'))
	#print("svs_dirs:",image_dir_list)
	for image_dir in image_dir_list :
		#print("image_dir:",image_dir)
		image_address = glob(join(image_dir, '*.svs'))
		if(len(image_address) == 0):
			#print("#####################image_address:", image_address)
			continue
		image_address_split = image_address[0].split("/")
		image_name = image_address_split[-1]
		#print("image_name:",image_name)
		label_name = 0
		#label_name = random.randint(0,1)
		label_name = get_label(image_name,label_address)
		if (label_name == -1):
			continue
		print("**************label_name:",label_name)
		##
		if need_save_WGI:
			#https://pypi.org/project/pyvips/2.0.4/
			img = pyvips.Image.new_from_file(image_address[0], access='sequential')
			WGI_address = image_address[0] + ".tiff"
			img.write_to_file(WGI_address)

		svs_file_slide = openslide.open_slide(image_address[0])
		image_sum += 1
		print("image_address[0]:",image_address[0])
		#print("svs_file_slide:", svs_file_slide.shape)
		need_deepzoom = False	
				
		#####choose region
		svs_address = image_address[0]
		region_list = choose_region(svs_file_slide, size_square, svs_address, save_model_address)
		image_data = region_list[0]
		#####choose region
		image_address_list.append(image_address)
		image_name_list.append(image_name)
		label_list.append(label_name)
		image_data_list.append(image_data)                
		
		#for svs_file in  glob(join(svs_dir, '*.svs')):
		#    print(svs_file)
	#get_label_list(image_name_list, label_address,label_list)
	#print("svs_file_list:", image_name_list)
	#print("labels_list:", label_list)
	return label_list , image_data_list
	'''temp comment
	#train_images = img_data.reshape((image_sum, image_size1, image_size2, image_size3))
	train_images = np.reshape(image_data_list, (image_sum, image_size1, image_size2, image_size3))

	#test_images = img_data.reshape((image_sum, image_size1, image_size2, image_size3))
	test_images = np.reshape(image_data_list, (image_sum, image_size1, image_size2, image_size3))

	train_images, test_images = train_images / 255.0, test_images / 255.0
	train_labels = label_list
	test_labels = label_list

	print("train_labels:",train_labels)
	model_train(train_images, train_labels, test_images, test_labels, image_size1, image_size2, image_size3, label_types)
	'''
