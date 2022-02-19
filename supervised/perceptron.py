"""
Perceptron assumptions:
    1. data is linearly separable
    2. binary classification problem ie y belongs to {-1, 1}

"""

import numpy as np
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self):
        self.W= None
        self.b= None
    
    def train(self, X, y):
        self.X_train= X
        self.y_train= y
        
        num_samples, num_features= self.X_train.shape
        
        self.W= np.zeros(num_features)
        self.b= np.zeros(num_samples)
        
        while True:
            counter= 0
            
            pred= np.dot(self.X_train, self.W)  + self.b
            pred= np.multiply(pred, y)
            
            #update only if sign(yW.Tx <= 0)
            update_idx= np.array([1 if i <= 0 else 0 for i in pred])
            t= update_idx * self.y_train 
            
            self.W += np.sum(np.dot(t, self.X_train), axis= 0)
            self.b += update_idx* self.y_train
            
            counter += np.sum(update_idx)
            
            if counter == 0:
                break
    
    def predict(self, X_test):
        l= np.dot(X_test, self.W)
        labels= [-1 if i <= 0 else 1 for i in l]
        
        return labels
    
    
if __name__== '__main__':
    
    X= np.array([[1, 5], [2, 3], [-1, -3], [-2, -5], [3, 6], [-3, -4]])
    y= np.array([1, 1, -1, -1, 1, -1])

    X_train, X_test, y_train, y_test= train_test_split(X, y)
    
    model= Perceptron()
    model.train(X_train, y_train)
    predictions= model.predict(X_test)
    accuracy= sum(predictions== y_test)/ y_test.shape[0]
