#coding=gbk
'''
Created on Oct 7, 2018

@author: ben
'''
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

#数据集地址
path='E:/AI/ctc/train_ctc/'

test_path='E:/AI/ctc/test_ctc/'
#模型保存地址
model_path='./log/model.ckpt'

#将所有的图片resize成100*100
w=100
h=100
c=3


#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    img_names=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img_names.append(im)
            #print('reading the images:%s'%(im))
            #print("reading the images: ",str(im).encode(encoding='utf_8', errors='ignore'))
            #print("reading the images: ",str(im).encode(encoding='utf_8', errors='strict'))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32),np.asarray(img_names)


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

#########prediction
def test(data_type):
    '''
    path1 = "E:/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
    path5 = "E:/flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
    path3 = "E:/flower_photos/roses/394990940_7af082cf8d_n.jpg"
    path4 = "E:/flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
    path2 = "E:/flower_photos/tulips/10791227_7168491604.jpg"
    '''
    '''
    path1 = "E:/AI/ctc/train/1/1.白玉.2.cep8.jpg"
    path2 = "E:/AI/ctc/train/0/0.白玉.1.cep8.jpg"
    path3 = "E:/AI/ctc/train/0/0.白玉.3.cep8.jpg"
    path4 = "E:/AI/ctc/train/1/1.白洁峰.1.cep8.jpg"
    path5 = "E:/AI/ctc/train/0/0.白玉.4.cep8.jpg"
    
        path1 = "E:/AI/ctc/test_ctc/1/闫兴秀.1.cep8.jpg"
    path2 = "E:/AI/ctc/test_ctc/1/闫秋均.1.cep8.jpg"
    path3 = "E:/AI/ctc/test_ctc/1/闫秋均.2.cep8.jpg"
    path4 = "E:/AI/ctc/test_ctc/1/魏勇.1.cep8.jpg"
    path5 = "E:/AI/ctc/test_ctc/1/许武珍.5.cep8.jpg"
    
    '''
    
    path1 = "E:/AI/ctc/test_ctc/1/王贵民.5.cep8.jpg"
    path2 = "E:/AI/ctc/test_ctc/1/王贵民.6.cep8.jpg"
    path3 = "E:/AI/ctc/test_ctc/1/王贵民.7.cep8.jpg"

    path4 = "E:/AI/ctc/test_ctc/1/杨建军.2.cep8.jpg"
    path5 = "E:/AI/ctc/test_ctc/1/杨建军.3.cep8.jpg"
    
    
    test_data,test_label,img_names=read_img(test_path)
    
    flower_dict = {0:'0',1:'1'}
    
    w=100
    h=100
    c=3
    
    def read_one_image(path):
        img = io.imread(path)
        img = transform.resize(img,(w,h))
        return np.asarray(img)
    
    with tf.Session() as sess:
        data = []
        
        if(data_type == 1):
            data1 = read_one_image(path1)
            data2 = read_one_image(path2)
            data3 = read_one_image(path3)
            data4 = read_one_image(path4)
            data5 = read_one_image(path5)
            data.append(data1)
            data.append(data2)
            data.append(data3)
            data.append(data4)
            data.append(data5)
        else:
            data = test_data
        
        
    
    
        saver = tf.train.import_meta_graph('./log/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./log/'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x:data}
    
        logits = graph.get_tensor_by_name("logits_eval:0")
    
        classification_result = sess.run(logits,feed_dict)
    
        #打印出预测矩阵
        print(classification_result)
        #打印出预测矩阵每一行最大值的索引
        print(tf.argmax(classification_result,1).eval())
        #根据索引通过字典对应花的分类
        output = []
        output = tf.argmax(classification_result,1).eval()
        for i in range(len(output)):
            #if(flower_dict[output[i]] == "1"):
            if True:
                #print("第",i+1,"朵花预测:"+flower_dict[output[i]])
                print(img_names[i]," ",flower_dict[output[i]])
            
    print("this is the end")

#train()
test(0)