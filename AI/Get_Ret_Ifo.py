import numpy as np
import cv2
import os
import openslide
from PIL import Image
from Slide_Window import slide_window


def get_ret_ifo(xy_list:list,slide,svs_address,window_size:int,stride:int,Points_Con_Thre:int,Area_Ratio_Thre:float):
    """From the xy_list ,getting the information which can help get a min circumscribed rectangle
    :param xy_list: 点的坐标列表，坐标以列表的形式表示
    :param slide:读取的svs文件
    :param num_name: 用于命名 第几张图片
    :param window_size:窗口大小
    :param stride:窗口步长
    :param Points_Con_Thre: 轮廓内点的个数阈值
    :param Area_Ratio_Thre: 面积阈值
    """

    for i in range(len(xy_list)):

        if len(xy_list[i]) == 0:
            continue
        #print(len(xy_list[i]))
        luncancer = 0
        health = 0
        img_regionall = []

        for points in xy_list[i]:
            if i==0:
                luncancer += 1
                print("   Deal with {0}th Lung Cancer area....".format(luncancer))
            if i==1:
                health += 1
                print("   Deal with Health area....")
            contours=np.array(points)
            x,y,w,h = cv2.boundingRect(contours)

            img_region = slide_window(slide,svs_address, x,y,w,h,window_size, stride,luncancer,health,i,contours,Points_Con_Thre,Area_Ratio_Thre)
            img_regionall.append(img_region)
    return img_regionall


if __name__=="__main__":

    get_ret_ifo(xy_list,slide,window_size,stride,Points_Con_Thre,Area_Ratio_Thre)
