import numpy as np
from collections import Counter
"""
Given a data point:
- Calculate it's distance from all other data points 
- Get the closest K points, K is a hyperparameter can be 2,3,5 etc

Classification: Gives the class with majority votes 
"""
def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predictions(self,X):
        #sending each point x in X to helper function to make predictions
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        #calculate the distance of each point x from X
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        
        #get closest k 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #majority votes 
        most_common = Counter(k_nearest_labels).most_common(self.k)
        return most_common[0][0]
        
        