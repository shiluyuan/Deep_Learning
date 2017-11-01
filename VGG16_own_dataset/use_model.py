import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import time
import numpy as np
from vgg16 import vgg16
from pred_image import pred_one_hot
from get_accuarcy import accuarcy
from PIL import Image

class img:
    def img_to_array(self):
        img_array = np.asarray(self.resize_image, dtype='float32')
        self.img_array = np.expand_dims(img_array, axis=0)
        
    def __init__(self,image):
        self.resize_image = image.resize((224,224))
        img.img_to_array(self)
        
img_1 = Image.open('A000041.jpg')
img_1 = img(img_1)

# load model
num_classes = 7 # num of your dataset
# VGG preprocessing
# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3])
softmax = vgg16(x, num_classes)
model = tflearn.DNN(softmax)

model.load("model.tflearn", weights_only=True)
model.predict(a.img_array)
#
