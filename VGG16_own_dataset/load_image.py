from tflearn.data_utils import image_preloader
import numpy as np

class load_image:
    def read_image(self):
        x, y = image_preloader(self.files_txt, image_shape=(224, 224), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)
        self.x_matrix = np.array(x)
        self.y_one_hot = np.array(y)
        return self.x_matrix, self.y_one_hot
    
    def one_hot_to_array(self):
        label_array = []
        for i in self.y_one_hot:
            max_index = np.argmax(i)
            label_array.append(max_index)
        self.label_array = label_array
        return self.label_array
    
    def __init__(self, files_txt):
        self.files_txt = files_txt
        load_image.read_image(self)
        load_image.one_hot_to_array(self)