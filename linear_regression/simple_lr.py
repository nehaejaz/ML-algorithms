class LinearRegression:
    def __init__(self):
        self.m = None 
        self.b = None 
        
    def fit(self, X_train, Y_train):
        num=0
        dem=0
        
        for i in range(len(X_train)):
            num += (X_train[i] - X_train.mean())*(Y_train[i] - Y_train.mean())
            dem += (X_train[i]-X_train.mean()) ** 2
            
        self.m = num/dem
        self.b = Y_train.mean() - self.m * X_train.mean()
    
    def predict(self,x_test):
        return self.m * x_test + self.b

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

X, y = datasets.load_diabetes(return_X_y=True)

#use only 1 feature 
X = X[:,np.newaxis,2]

#split the data
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=2)

lr = LinearRegression()
print(len(X_train),len(y_train))
lr.fit(X_train,y_train)
prediction = lr.predict(X_test[0])
print(f'Predicted={prediction}')
print(X_test[0],y_test[0])




