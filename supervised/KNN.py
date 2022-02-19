"""
KNN assumptions:
    1. similar points will have same labels
    2. data has low intrinsic dimentionality
    
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN():
    def __init__(self, k):
        self.k= k
        
    def train(self, X, y):
        self.X_train= X
        self.y_train= y
    
    def compute_distances(self, X_test):
        #(a-b)^2= a^2 - 2a.b + b^2
        X_test_squared= np.sum(X_test**2, axis=1, keepdims= True)   # (n_test, d)-> (n_test, 1)
        
        X_train_squared= np.sum(self.X_train**2, axis=1, keepdims= True)    ## (n_train, d)-> (n_train, 1)
        
        dot_product= np.dot(X_test, self.X_train.T)  #(n_test, n_train)
        
        #(n_test, 1) + (n_test, n_train) + (1, n_train) -> (n_test, n_train)
        return np.sqrt(X_test_squared + X_train_squared.T - 2*dot_product)
    
    def predict(self, X_test):
        distances= self.compute_distances(X_test)
        return self.predict_labels(distances)
    
    def predict_labels(self, distances):
        num_test= X_test.shape[0]
        y_pred= np.zeros(num_test, dtype= 'int32')
        
        for i in range(num_test):
            #sort pts. by their dist
            y_idx= np.argsort(distances[i, :])
            
            #get labels of k closest points
            k_closest= self.y_train[y_idx[:self.k]]
            
            #count for each label, return the one with highest count
            y_pred[i]= np.argmax(np.bincount(k_closest))
            
        return y_pred
    
    
if __name__== '__main__':
    X, y= datasets.make_blobs(n_samples= 500, n_features= 2, centers= 6)
    X_train, X_test, y_train, y_test= train_test_split(X, y)
    
    model= KNN(k=5)
    model.train(X_train, y_train)
    predictions= model.predict(X_test)
    accuracy= sum(predictions== y_test)/ y_test.shape[0]
