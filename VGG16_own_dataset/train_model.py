import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import time
import numpy as np
from vgg16 import vgg16
from load_image import load_image
from pred_image import pred_one_hot
from get_accuarcy import accuarcy
# load the data
train_files_txt = "train_fvgg_emo.txt"
test_files_txt = "test_fvgg_emo.txt"

train= load_image(train_files_txt)
train_x = train.x_matrix
train_y = train.y_one_hot

test = load_image(test_files_txt)
test_x = test.x_matrix
test_y = test.y_one_hot

print("train_x:",train_x.shape)
print("train_y:",train_y.shape)

# import vgg16
num_classes = 7 # num of your dataset
# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
                                     per_channel=True)
# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16(x, num_classes)
regression = tflearn.regression(softmax, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.0001, restore=False)
model = tflearn.DNN(regression)

model.load("vgg16.tflearn", weights_only=True)

# Start finetuning
time_0 = time.time()
model.fit(train_x, train_y, n_epoch=20, validation_set=0.1, shuffle=False,
          show_metric=True, batch_size=32, snapshot_epoch=False,
          snapshot_step=200, run_id='vgg-finetuning')
print("total time is:", time.time() - time_0)

## check predict accuarcy
pred_test_y_one_hot = pred_one_hot(model, test_x , 50)
acc = accuarcy(pred_test_y_one_hot, test_y)
print("test accuarcy:",acc.acc)

# at last we save the model
model.save('model.tflearn')







