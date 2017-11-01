import numpy as np

def pred_one_hot(model, x , batch_size):
    predict_one_hot = []
    total_batch = int(len(x) / batch_size)
    
    for i in range(total_batch):
        temp_pred = model.predict(x[i*batch_size: (i+1)*batch_size])
        predict_one_hot += temp_pred.tolist()
    
    predict_one_hot += model.predict(x[(i+1)*batch_size:]).tolist()   
    
    return np.array(predict_one_hot)