import cv2


def judge_position(contours,point_list):
    '''去除轮廓以外选中的干扰区域
    :param contours: 轮廓
    :param point_list: 滑窗的顶点和中心坐标列表
    :return: 是否在轮廓中的状态列表 1在轮廓中 -1 不在轮廓中
    '''
    value_list=[]
    for point in point_list:
        #print(point)
        value=cv2.pointPolygonTest(contours,point,False)
        value_list.append(value)
    #print(value_list)
    return value_list


if __name__=="__main":

    judge_position(contours, point_list)
