# -*- coding: utf-8 -*-  
'''
Created on Sep 26, 2018

@author: ben
'''
#-*- coding: UTF-8 -*- 
from bokeh.util.paths import ROOT_DIR

'''
Author: Steve Wang
Time: 2017/12/8 10:00
Environment: Python 3.6.2 |Anaconda 4.3.30 custom (64-bit) Opencv 3.3
'''

import cv2
import numpy as np
import os


def get_image(path):
    #��ȡͼƬ
    #path_code = path.
    #img=cv2.imread(path)
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    if img.all() == None:
        print ("--------------------------------look here")
        return None,None
    print(" path in get_image: ", path)


    return img

def get_gray(img):
    #��ȡͼƬ
    #path_code = path.
    #img=cv2.imread(path)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return gray

def Gaussian_Blur(gray):
    # ��˹ȥ��
    blurred = cv2.GaussianBlur(gray, (5, 5),0)

    return blurred

def Sobel_gradient(blurred):
    # ���ȶ�����������x��y�����ݶ�
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

def Thresh_and_blur(gradient):

    blurred = cv2.GaussianBlur(gradient, (5,5),0)
    #(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    (_, thresh) = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    return thresh

def image_morphology(thresh):
    # ����һ����Բ�˺���
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    #print(" kernel is",kernel)
    # ִ��ͼ����̬ѧ, ϸ��ֱ�Ӳ��ĵ����ܼ�
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 
    closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) 
    #closed = cv2.erode(closed, kernel, iterations=2)
    #closed = cv2.dilate(closed, kernel, iterations=4)
    

    return closed

def findcnts_and_box_point(closed):
    # ����opencv3���ص�����������
    (_, cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    #print (" cnts: ",cnts)
    sorted_list = sorted(cnts, key=cv2.contourArea, reverse=True)
    box_list = []
    if(len(sorted_list)>0):
        for list_index in sorted_list:
            c = list_index
            rect = cv2.minAreaRect(c)
            box_list.append(np.int0(cv2.boxPoints(rect)))
    #c = sorted(cnts, key=cv2.contourArea, reverse=True)[2]
    # compute the rotated bounding box of the largest contour
    #rect = cv2.minAreaRect(c)
    #box = np.int0(cv2.boxPoints(rect))

    return box_list

def drawcnts_and_cut(original_img, box_list):
    draw_img = original_img.copy()
    crop_img_list = []
    for box in box_list:
        # ��Ϊ��������м�ǿ���ƻ��ԣ�������Ҫ��img.copy()�ϻ�
        # draw a bounding box arounded the detected barcode and display the image
        draw_img = cv2.drawContours(draw_img, [box], -1, (0, 0, 255), 3)
    
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = original_img[y1:y1+hight, x1:x1+width]
        print("x1",x1,"x2",x2,"y1",y1,"y2",y2,"hight",hight,"width",width)
        crop_img_list.append(crop_img)
    return draw_img, crop_img_list


def image_process(img,save_path_draft):
    #original_img, gray = get_image(img)
    original_img =img
    gray = get_gray(img)
    if original_img.all() ==None:
        return
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    gradient = blurred
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box_list = findcnts_and_box_point(closed)
    #for box in box_list: 
    if True:     
        draw_img, crop_img_list = drawcnts_and_cut(original_img,box_list)
    
        # ����һ�㣬�����Ƕ���ʾ��������
    
        #cv2.imshow('original_img', original_img)
        
        #cv2.namedWindow('blurred',0)
        #cv2.imshow('blurred', blurred)
        
        #cv2.imencode('.jpg',blurred)[1].tofile(save_path_draft + "blurred.jpg")
        
        
        #cv2.namedWindow('gradX',0)
        #cv2.imshow('gradX', gradX)
        
        #cv2.imencode('.jpg',gradX)[1].tofile(save_path_draft + "gradX.jpg")
        #cv2.namedWindow('gradY',0)
        #cv2.imshow('gradY', gradY)
        
        #cv2.imencode('.jpg',gradY)[1].tofile(save_path_draft + "gradY.jpg")
                
        #cv2.namedWindow('final',0)
        #cv2.imshow('final', gradient)
        
        # cv2.imencode('.jpg',gradient)[1].tofile(save_path_draft + "gradient.jpg")
         
        #cv2.namedWindow('thresh',0)
        #cv2.imshow('thresh', thresh)
        
        # cv2.imencode('.jpg',thresh)[1].tofile(save_path_draft + "thresh.jpg")
        
        
        #cv2.namedWindow('closed',0)
        #cv2.imshow('closed', closed)
        
        #cv2.imencode('.jpg',closed)[1].tofile(save_path_draft + "closed.jpg")
        #cv2.waitKey(100000)

    return draw_img, box_list, crop_img_list

def walk():
    save_path_small_img = 'E:/AI/ctc/small_image/'
    save_path_orignal = 'E:/AI/ctc/small_image_orignal/'
    save_path_orignal_0 = 'E:/AI/ctc/0/'
    save_path_orignal_1 = 'E:/AI/ctc/1/'
    save_path_big_img = "E:/AI/ctc/big_image/"
    
    if not os.path.exists(save_path_small_img):
        os.makedirs(save_path_small_img) 

    if not os.path.exists(save_path_orignal):
        os.makedirs(save_path_orignal) 
        
    if not os.path.exists(save_path_orignal_0):
        os.makedirs(save_path_orignal_0)         
        
    if not os.path.exists(save_path_orignal_1):
        os.makedirs(save_path_orignal_1) 

    if not os.path.exists(save_path_big_img):
        os.makedirs(save_path_big_img) 

    
    filename_rgb = r'E:/AI/ctc/CTC/'
    for filename in os.listdir(filename_rgb):              #listdir�Ĳ������ļ��е�·��
        print (str(filename))
    
    #for filename1 in glob.glob(r'E:/AI/ctc/CTC/*.exe'):
    #    print (filename1)
    ROOT_DIR_CTC = 'E:/AI/ctc/CTC/'
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR_CTC):
        print ('Directory', dirpath, '  dirnames: ',dirnames)
        
        for filename2 in filenames:
            Last_path = dirpath.lstrip(ROOT_DIR_CTC)
            Last_path = Last_path
            #print (' File', filename2)
            if(filename2 == "dapi.jpg"):
                img_path = dirpath
                filename_dapi = "dapi"
                filename_dapi_jpg = "dapi.jpg"
                filename_cep8 = "cep8" 
                filename_cep8_jpg = "cep8.jpg"
                dapi_img_path =  dirpath +'/'+ filename_dapi_jpg
                cep8_img_path =   dirpath +'/'+ filename_cep8_jpg
                #save_path = 'E:\\AI\\ctc\\result\\result_of_'

                save_path_big_img_custom_dapi_jpg = "E:/AI/ctc/big_image/" + Last_path + filename_dapi_jpg
                save_path_draft_big = "E:/AI/ctc/big_image/"
                ######确定文件后
                img =  get_image(dapi_img_path)
                
                try:
                    draw_img, box_list, crop_img_list = image_process(img, save_path_big_img + Last_path + filename_dapi)
                
                except:
                    continue
                ##########
                #cv2.namedWindow('draw_img',0)
                #cv2.imshow('draw_img', draw_img)
                #cv2.imwrite(save_path, draw_img)
                cv2.imencode('.jpg',draw_img)[1].tofile(save_path_big_img_custom_dapi_jpg)
                '''
                number = 0
                print (" len of crop image: ",str(len(crop_img_list)))
                for crop_img in crop_img_list:
                    number = number + 1
                    crop_img_name = 'crop_img' + str(number)
                    cv2.namedWindow(crop_img_name,0)
                    cv2.imshow(crop_img_name, crop_img)
                    #
                    #cv2.imwrite(save_path, crop_img)
                '''
                #cv2.waitKey(20171219)
                try:
                    cep8_img =  get_image(cep8_img_path)
                except:
                    continue
                draw_cep8_img, crop_cep8_img_list = drawcnts_and_cut(cep8_img,box_list)
                cep8_img_num = 0
                for crop_cep8_img in crop_cep8_img_list:
                    cep8_img_num = cep8_img_num + 1
                    save_path_cep8_img = save_path_small_img + Last_path + "."+ str(cep8_img_num) + "." + filename_cep8_jpg
                    print(" image_process in image:", save_path_cep8_img)
                    try:
                        draw_img1, box_list1, crop_img_list1 = image_process(crop_cep8_img,save_path_small_img + Last_path + "."+ str(cep8_img_num) + "." + filename_cep8 )
                    except:
                        continue
                    #save drawed small image
                                       
                    cv2.imencode('.jpg',draw_img1)[1].tofile(save_path_cep8_img)
                    
                    #save orignal small image
                    save_path_cep8_img_orignal = save_path_orignal + Last_path + "."+ str(cep8_img_num) + "." + filename_cep8_jpg       
                    save_path_cep8_img_orignal_0 = save_path_orignal_0 + Last_path + "."+ str(cep8_img_num) + "." + filename_cep8_jpg     
                    save_path_cep8_img_orignal_1 = save_path_orignal_1 + Last_path + "."+ str(cep8_img_num) + "." + filename_cep8_jpg                  
                    cv2.imencode('.jpg',crop_cep8_img)[1].tofile(save_path_cep8_img_orignal)                
                    if len(box_list1)>2:
                        cv2.imencode('.jpg',crop_cep8_img)[1].tofile(save_path_cep8_img_orignal_1)
                    else:
                        cv2.imencode('.jpg',crop_cep8_img)[1].tofile(save_path_cep8_img_orignal_0)
                ##########
                
                
                #######
                
                
    print ("this is the end")
            
walk()

