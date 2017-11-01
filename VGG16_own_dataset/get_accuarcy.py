import numpy as np

class accuarcy:        
    def one_hot_to_array(self):
        label_array = []
        for i in self:
            max_index = np.argmax(i)
            label_array.append(max_index)
        return label_array
    
    def get_array(self):
        self.pred_y_array = accuarcy.one_hot_to_array(self.pred_y_one_hot)
        self.true_y_array = accuarcy.one_hot_to_array(self.true_y_one_hot)
        
    
    def get_accuarcy(self):
        total = 0
        for i in range(len(self.true_y_one_hot)):
            if self.true_y_array[i] == self.pred_y_array[i]:
                total += 1
        self.acc = total/len(self.true_y_array)
    
    def __init__(self, pred_y_one_hot,true_y_one_hot):
        self.true_y_one_hot = true_y_one_hot
        self.pred_y_one_hot = pred_y_one_hot
        accuarcy.get_array(self)
        accuarcy.get_accuarcy(self)


a = np.array([[0,1],[0,1]])
b = np.array([[1,0],[0,1]])
c = accuarcy(a,b)
