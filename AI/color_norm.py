import numpy as np
import os
import scipy.misc as misc
from ops import find_files
from os.path import join, split
from model_color import DCGMM
from get_config import get_config
from read import SampleProvider
from ops import image_dist_transform
import ops as utils
from glob import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import openslide
#

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "prediction", "Mode train/ prediction")
tf.flags.DEFINE_string("logs_dir", "/data2/ben/HE/data/result/color_logs/", "path to logs directory")
tf.flags.DEFINE_integer("step_x", 500, "step_y")
tf.flags.DEFINE_integer("step_y", 500, "step_y")
tf.flags.DEFINE_string("save_model_address", "../../data/result/my_model.CNN_3_reg-tmb1", "save_model_address")
tf.flags.DEFINE_integer("begin_x", 43000, "begin_x")
tf.flags.DEFINE_integer("begin_y", 49000, "begin_y")
tf.flags.DEFINE_string("images_address", "../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs", "images_address")

def color(data,template,logs_dir="/data2/ben/HE/data/result/color_logs/"):
    import tensorflow.compat.v1 as tf
    tf.reset_default_graph()
    sess = tf.Session()
    image_dist = []
    is_train = False
    #print("FLAGS",tf.flags.FLAGS)
    #print("FLAGS.logs_dir==============",tf.flags.FLAGS.logs_dir)
    config = get_config(FLAGS, is_train)
    dist = DCGMM(sess, config, "DCGMM", is_train)

    db_tmpl = SampleProvider("Template_dataset", template, config.fileformat, config.image_options, is_train)

    mu_tmpl = 0
    std_tmpl = 0
    N = 0
    while True:
        X = db_tmpl.DrawSample(config.batch_size)

        if len(X) == 0:
            break
        X_hsd = utils.RGB2HSD(X / 255.0)

        mu, std, gamma = dist.deploy(X_hsd)
        mu = np.asarray(mu)
        mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
        std = np.asarray(std)
        std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]

        N = N + 1
        mu_tmpl = (N - 1) / N * mu_tmpl + 1 / N * mu
        std_tmpl = (N - 1) / N * std_tmpl + 1 / N * std


    db = SampleProvider("Test_dataset", data, config.fileformat, config.image_options, is_train)

    while True:
        X = db.DrawSample(config.batch_size)
        if len(X) == 0:
            break

        X_hsd = utils.RGB2HSD(X / 255.0)
        mu, std, pi = dist.deploy(X_hsd)
        mu = np.asarray(mu)
        mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
        std = np.asarray(std)
        std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
        X_conv = image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, config.im_size, config.ClusterNo)
        image_dist.append(X_conv)
       # misc.imsave('/data2/ben/HE/data/result/color_tmp/color_norma.png', np.squeeze(X_conv))

    return image_dist

# if __name__ == '__main__':
#     print("###########################################this is beginning: \n")
#     svs_file_slide = openslide.open_slide("/data2/practice/yezx/cecs/f80072d9-4a58-44f4-be9d-90117eeabbe6/154.png")
#     image_data = svs_file_slide.read_region(( 0,0), 0,(512,512)).convert('RGB')
#     images_tmp = np.array([misc.imread("/data2/ben/HE/data/result/color_tmp/tmp.orig.png")])
#     test_image = np.reshape(image_data, (1, 512, 512, 3))
#     color_test_image = color(test_image, images_tmp)
#     print(color_test_image[0].shape)

