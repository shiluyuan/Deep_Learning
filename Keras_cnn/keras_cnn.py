from __future__ import print_function
import pandas as pd
from pandas import DataFrame
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")
train = train.sample(frac=1).reset_index(drop=True)
# define
img_rows, img_cols = 28, 28
n_classes = 10
batch_size = 128
trainning_period = 20
learning_rate = 1e-3
# format the data
train_x = train.iloc[:,1:].as_matrix().reshape(train.shape[0], img_rows, img_cols)
train_y_array = train.iloc[:,0]
train_y = keras.utils.to_categorical(train_y_array, n_classes)

test_x = test.iloc[:,1:].as_matrix().reshape(test.shape[0], img_rows, img_cols)
test_y_array = test.iloc[:,0]
test_y = keras.utils.to_categorical(test_y_array, n_classes)

if K.image_data_format() == 'channels_first':
    train_x = train_x.reshape(train_x.shape[0], 1, img_rows, img_cols)
    test_x = test_x.reshape(test_x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)
    test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# build network
model = Sequential()
# add first convolutional layer
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',padding='valid',
                 input_shape=input_shape))
# add maxpooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# add second convolutional layer
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',padding='valid',
                 input_shape=input_shape))
# add maxpooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# add a fully connected layer
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
# add a dropout rate
model.add(Dropout(0.2))
# add final layer 
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(lr=learning_rate,decay=0.01),
              metrics=['accuracy'])
# fit the model
time_0 = time.time()
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=trainning_period,
          verbose=1,
          validation_data=(test_x, test_y))
print("total time is:", time.time() - time_0)
# predict probablity of test data
pred_prob = model.predict_proba(test_x)
print(pred_prob)
# predict class of test data
pred_class = model.predict_classes(test_x)
print(pred_class)







