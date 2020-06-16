import numpy as np
import scipy.misc as misc
from scipy.ndimage import rotate
from ops import find_files
import os
import math
import cv2
from PIL import  ImageEnhance
from glob import glob
from os.path import join, split
import openslide
class SampleProvider(object):
  
  def __init__(self, name, data_dir, fileformat, image_options, is_train):
    self.name = name
    self.is_train = is_train
    self.path = data_dir
    self.fileformat = fileformat
    self.reset_batch_offset()
    self.files,self.dir_list,self.file_name = self._create_image_lists()
    self.image_options = image_options
    self._read_images()
    
  def _create_image_lists(self):
    if not os.path.exists(self.path):    
        print("Image directory '" + self.path + "' not found.")
        return None

    if self.is_train == False:
        file_list1 = list()
        file_name1 = list()
        #for filename in find_files(self.path, '*.orig.' + self.fileformat):
        for filename in find_files(self.path, '*.' + self.fileformat):
            file_list1.append(filename)
            image_address_split = filename.split("/")
            image_name = image_address_split[-1]
            file_name1.append(image_name)

        print('No. of files: %d' % (len(file_list1)))
        return file_list1,self.path,file_name1
    
    file_list = list()
    file_name= list()
    image_dir_list = glob(join(self.path , r'*/'))
    for image_dir in image_dir_list:

        image_address = glob(join(image_dir, '*orig.png'))
        if (len(image_address) == 0):
            continue
        # for i in range(len(image_address)):
        for i in range(len(image_address)):
            if (len(image_address) < i + 1):
                continue
            image_address_split = image_address[i].split("/")
            image_name = image_address_split[-1]
            file_name.append(image_name)
            file_list.append(image_address[i])

    print ('No. of files: %d' % (len(file_list)))
    return file_list,image_dir_list,file_name

  def _read_images(self):
    self.__channels = True
    self.images_org = np.array([misc.imread(filename)for filename in self.files])
    print("############################")
    # self.annotations = np.array(
    #     [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
   
  def _transform(self, images_org):
      global image_new
      if self.image_options["resize"]:
          resize_size = int(self.image_options["resize_size"])
          image_new = misc.imresize(images_org, [resize_size, resize_size], interp='nearest')

      if self.image_options["flip"]:
          if (np.random.rand() < .5):
              image_new = cv2.flip(images_org, 0)
          else:
              image_new = cv2.flip(images_org, 1)

      if self.image_options["rotate_stepwise"]:
              if(np.random.rand()>.25): # skip "0" angle rotation
                  angle = int(np.random.permutation([1,2,3])[0] * 90)
                  image_new = rotate(images_org, angle, reshape=False)
      if self.image_options["environment factor"]:
          hsv = cv2.cvtColor(images_org, cv2.COLOR_BGR2HSV)  # 增加饱和度光照的噪声
          hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
          hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
          hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)
          image_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      else:
          image_new = images_org

      return np.array(image_new)
  #
  # def get_records(self):
  #   return self.files, self.annotations
        
  def get_records_info(self):
      #返回图片和标签全路径
      return self.files
        
  def reset_batch_offset(self, offset=0):
      self.batch_offset = offset
      self.epochs_completed = 0

  def DrawSample(self, batch_size):
    start = self.batch_offset
    self.batch_offset += batch_size
    if self.batch_offset > self.images_org.shape[0]:
        
        if not self.is_train:
            image = []
            return image
            
        # Finished epoch
        self.epochs_completed += 1
        print(">> Epochs completed: #" + str(self.epochs_completed))
        # Shuffle the data
        perm = np.arange(self.images_org.shape[0], dtype=np.int)
        np.random.shuffle(perm)
        
        self.images_org = self.images_org[perm]
        self.files = [self.files[k] for k in perm] 
        
        # Start next epoch
        start = 0
        self.batch_offset = batch_size

    end = self.batch_offset
    
  
    image = [self._transform(self.images_org[k]) for k in range(start,end)]
    curfilename = [self.files[k] for k in range(start, end)]

    return np.asarray(image), curfilename


