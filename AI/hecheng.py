#!/data2/ben/anaconda3/envs/python35/bin/python
import openslide
import numpy as np
import imageio

image1 = openslide.open_slide("/data2/ben/data/geneis/CTC/temp/cep17.jpg")

dim0 = image1.dimensions[0]
dim1 = image1.dimensions[1]

image1_data = image1.read_region((0, 0), 0, (dim0,dim1)).convert('RGB')


image2 = openslide.open_slide("/data2/ben/data/geneis/CTC/temp/cep8.jpg")
image2_data = image2.read_region((0, 0), 0, (dim0,dim1)).convert('RGB')



image3 = openslide.open_slide("/data2/ben/data/geneis/CTC/temp/dapi.jpg")
image3_data = image3.read_region((0, 0), 0, (dim0,dim1)).convert('RGB')


image4 = openslide.open_slide("/data2/ben/data/geneis/CTC/temp/red.jpg")
image4_data = image4.read_region((0, 0), 0, (dim0,dim1)).convert('RGB')


#image_compound= ( np.array(image1_data) +  np.array(image2_data) +  np.array(image3_data))/3
image_compound=  np.array(image1_data) * 0.2 +  np.array(image2_data) * 0.2 + np.array(image4_data)*0.2 +  np.array(image3_data)*0.4




#image_compound.save("/data2/ben/data/geneis/CTC/temp/compound.jpg")

imageio.imsave("/data2/ben/data/geneis/CTC/temp/compound.jpg", image_compound)
