from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X,y = datasets.load_diabetes(return_X_y=True)

#select only 1 input feature
X = X[:, np.newaxis, 2]

#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2, test_size=0.2, )

#Apply Lr 

lr = LinearRegression()
lr.fit(X_train,y_train)
prediction = lr.predict(X_test[0].reshape(1,1))
print(f'Predicted={prediction}')
print(X_test[0],y_test[0])

m = lr.coef_
b= lr.intercept_
print(f"m={m}, b={b}")

plt.scatter(X,y)
plt.plot(X_train,lr.predict(X_train), color="red")
plt.show()