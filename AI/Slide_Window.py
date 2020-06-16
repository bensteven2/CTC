import cv2
import numpy as np
from Judge_Position import judge_position
from Get_Area_Ratio import get_area_ratio


def slide_window(image,svs_address,x_begin:int,y_begin:int,w:int,h:int,window_size:int, stride:int,luncancer:int,health:int,
                 i,contours,Points_Con_Thre,Area_Ratio_Thre):
    '''通过滑窗的方式得到小窗口
    :param image: svs文件
    :param x_begin: 滑窗的左上角坐标x
    :param y_begin: 滑窗的左上角坐标y
    :param w: 外接矩形的宽
    :param h: 外接矩形的高
    :param window_size: 滑窗大小
    :param stride: 滑窗步长
    :param num_name: 图片索引
    :param luncancer: 癌症区域索引
    :param health: 健康区域索引
    :param i: 癌症类型标志
    :param contours: 轮廓
    :param Points_Con_Thre: 轮廓内点的个数阈值
    :param Area_Ratio_Thre: window内面积比率阈值
    '''
    # 统计一个区域得到小窗口的个数
    i_name=0
    img_resion = []

    # 异常控制
    if w<window_size:
        w=window_size+10
    if h<window_size:
        h=window_size+10

    [m, n] = image.dimensions
    x_end=x_begin+w-window_size
    y_end=y_begin+h-window_size
    for x in range(x_begin, x_end, stride):
        for y in range(y_begin,y_end , stride):

           i_name+=1


           # 越界控制
           if x+window_size>m or y+window_size>n:
               continue

           # 去除轮廓外干扰区域
           point_list=[(x+int(window_size/2),y+int(window_size/2)),(x,y),(x+window_size,y),(x,y+window_size),(x+window_size,y+window_size)]
           #print("point_list:",point_list)
           count_list=judge_position(contours,point_list)
           if count_list.count(1.0)<Points_Con_Thre or count_list[0]==-1.0:
               continue

           #ret = np.array(image.read_region((x, y), 0, (window_size, window_size)))

           ret = image.read_region((x, y), 0, (window_size,window_size)).convert('RGB')


           # 去除轮廓内的白色
           ratio=get_area_ratio(ret)

           if i == 0:
               if ratio>Area_Ratio_Thre:
                   continue
               print("      get {0}th LungCancer picture".format(i_name))
               img_resion.append(ret)
               ret.save(svs_address + "_" + str(luncancer) + "_" + str(i_name) + "orig.png")


           # if i == 1:
           #     print("      get {0}th Health picture".format(i_name))
           #     ret.save("E:/tcga/result/Health/" + str(num_name) + "_" + str(health) + "_" + str(i_name) + ".png")
    return img_resion


if __name__=="__main__":

    slide_window(image,x_begin,y_begin,w,h,window_size, stride,num_name,luncancer,health,i,contours,Points_Con_Thre,Area_Ratio_Thre)


