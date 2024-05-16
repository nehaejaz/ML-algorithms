from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X,y = datasets.load_diabetes(return_X_y=True)

#select only 1 input feature
X = X[:, np.newaxis, 2]

#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2, test_size=0.2, )

#Apply Lr 

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

m = lr.coef_
b= lr.intercept_
print(f"m={m}, b={b}")

print(f"MAE is {mean_absolute_error(y_test,y_pred)}")
print(f"MSE is {mean_squared_error(y_test,y_pred)}")
print(f"RMSE is {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"R2 is {r2_score(y_test,y_pred)}")

# plt.scatter(X,y)
# plt.plot(X_train,lr.predict(X_train), color="red")
# plt.show()