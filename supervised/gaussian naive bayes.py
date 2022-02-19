"""
Gaussian naive bayes assumptions:
    1. given the label, the features of the input vector are independent of each other
    2. each feature is countinuous and belongs to a gaussian distribution
    
"""

import numpy as np
from sklearn.model_selection import train_test_split

# class GaussianNB:
#     def __init__():
#         pass
    
#     def train(X_train, y_train):
#         labels= np.unique(y_train)
#         prob_y= np.zeros(labels.shape)
        
#         for i in labels:
            