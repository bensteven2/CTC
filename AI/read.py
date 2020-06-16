import numpy as np
import scipy.misc as misc
from scipy.ndimage import rotate
from ops import find_files
import os
import math
import cv2
from PIL import ImageEnhance
from glob import glob
from os.path import join, split
import openslide


class SampleProvider(object):

    def __init__(self, name, image, fileformat, image_options, is_train):
        self.name = name
        self.data = image
        self.is_train = is_train
        self.fileformat = fileformat
        self.reset_batch_offset()
        self.image_options = image_options
        self._read_images()


    def _read_images(self):
        self.__channels = True

        self.images_org =  self.data
        print(self.images_org)
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
            if (np.random.rand() > .25):  # skip "0" angle rotation
                angle = int(np.random.permutation([1, 2, 3])[0] * 90)
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

            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        image = [self._transform(self.images_org[k]) for k in range(start, end)]

        return np.asarray(image)
