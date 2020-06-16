''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''


# import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
from ops import find_files
from os.path import join, split
from model_color import DCGMM
from config import get_config
from Sample_Provider import SampleProvider
from ops import image_dist_transform
import ops as utils
from glob import glob
import tensorflow.compat.v1 as tf
import argparse


def main():
  sess = tf.Session()
  image_dist = []
  if FLAGS.mode == "train": 
      is_train = True
  else:
      is_train = False
      
  config = get_config(FLAGS, is_train)
  # if not os.path.exists(config.logs_dir):
  #     os.makedirs(config.logs_dir)
    
  dist = DCGMM(sess, config, "DCGMM", is_train)

  
  if FLAGS.mode == "train":
      
      db = SampleProvider("Train_dataset", config.data_dir, config.fileformat, config.image_options, is_train)
      for i in range(int(config.iteration)):
        X = db.DrawSample(config.batch_size)
        X_hsd = utils.RGB2HSD(X[0]/255.0)
        loss, summary_str, summary_writer = dist.fit(X_hsd)
        
        if i % config.ReportInterval == 0:
            summary_writer.add_summary(summary_str, i)
            print("iter {:>6d} : {}".format(i+1, loss))
            
        if i % config.SavingInterval == 0:
            dist.saver.save(sess, config.logs_dir+ "model.ckpt", i)
        
  elif FLAGS.mode == "prediction":  
    
      # if not os.path.exists(config.out_dir):
      #     os.makedirs(config.out_dir)
     
      db_tmpl = SampleProvider("Template_dataset", config.tmpl_dir, config.fileformat, config.image_options, is_train)
      print("db_tmpl")
      print(db_tmpl)
      mu_tmpl = 0
      std_tmpl = 0
      N = 0
      while True:
          X = db_tmpl.DrawSample(config.batch_size)
          
          if len(X) ==0:
              break
          
          X_hsd = utils.RGB2HSD(X[0]/255.0)
          
          mu, std, gamma = dist.deploy(X_hsd)
          mu = np.asarray(mu)
          mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
          std = np.asarray(std)
          std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]
          
          N = N+1
          mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
          std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
      
      print("Estimated Mu for template(s):")
      print(mu_tmpl)
      
      print("Estimated Sigma for template(s):")
      print(std_tmpl)



      image_dir_list = glob(join(config.data_dir, r'*/'))
      for image_dir in image_dir_list:
          print(image_dir)
          png_addr = glob(join(image_dir, '*color_norma.png'))
          if png_addr != []:
              continue
          db = SampleProvider("Test_dataset", image_dir, config.fileformat, config.image_options, is_train)

          while True:
              X = db.DrawSample(config.batch_size)
              if len(X) ==0:
                  break

              X_hsd = utils.RGB2HSD(X[0]/255.0)
              mu, std, pi = dist.deploy(X_hsd)
              mu = np.asarray(mu)
              mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
              std = np.asarray(std)
              std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]
              X_conv = image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, config.im_size, config.ClusterNo)
              image_dist.append(X_conv)

              filename = X[1]
              # filename = filename[0].split('/')[-1]
              # if not os.path.exists(config.out_dir):
              #    os.makedirs(config.out_dir)
              misc.imsave(filename[0]+'_color_norma.png', np.squeeze(X_conv))
              #misc.imsave(config.out_dir + filename, np.squeeze(X_conv))
      print("######################")
      print(len(image_dist))

  else:
      print('Invalid "mode" string!')
      return 

if __name__ == "__main__":

  print("###########################################this is beginning: \n")
  parser = argparse.ArgumentParser(description='manual to this script', epilog="authors of this script are PengChao YeZixuan XiaoYupei and Ben ")
  parser.add_argument('--log_dir', type=str, default = "/data2/ben/HE/data/result/color_logs/")
  parser.add_argument('--data_dir', type=str, default = "/data2/practice/yezx/cecs/")
  parser.add_argument('--tmpl_dir', type=str, default = "/data2/ben/HE/data/result/color_tmp/")
  args = parser.parse_args()


  tf.disable_v2_behavior()
  FLAGS = tf.flags.FLAGS
  tf.flags.DEFINE_string('mode', "prediction", "Mode train/ prediction")
  tf.flags.DEFINE_string("logs_dir", args.log_dir, "path to logs directory")
  tf.flags.DEFINE_string("data_dir", args.data_dir, "path to dataset")
  tf.flags.DEFINE_string("tmpl_dir", args.tmpl_dir, "path to template image(s)")
  #tf.flags.DEFINE_string("out_dir", "/data2/ben/HE/data/result/color_out", "path to template image(s)")
  main()
