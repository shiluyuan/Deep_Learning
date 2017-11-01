import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
def vgg16(input, num_class):

    #in the model, we added trainable=False to make sure the parameter are not updated during training
    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1',trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2',trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1',trainable=False)
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2',trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1',trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2',trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3',trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1',trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2',trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3',trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')
    #we changed the structure here to let the fc only have 2048, less parameter, enough for our task
    x = tflearn.fully_connected(x, 2048, activation='relu', scope='fc7',restore=False)
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)

    return x
