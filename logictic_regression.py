from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 

#Tahing a look at the data features
data = datasets.load_breast_cancer()
# print(data.feature_names)

X, y = datasets.load_breast_cancer(return_X_y=True)


#Select only 1 input feature
# X = X[:, np.newaxis, 2]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2, test_size=0.2)

#Analysixzing the behaviour in data
# plt.scatter(X,y)
# plt.show()

#Training the model 
logist_reg = LogisticRegression()
logist_reg.fit(X_train,y_train)
y_predi = logist_reg.predict(X_test)
print(y_test[0],y_predi[0])

#Checking the accuracy of the model on test set 
accuracy = logist_reg.score(X_test,y_test)
print(f'The accuracy is {accuracy}')

'''
The accuracy is 86 percent
'''

#Predic the probabilities of each input data in test
probability = logist_reg.predict_proba(X_test)
print(f'The probability is {probability[0]}')

'''
The probability is [0.28996281 0.71003719] which shows 71 percent there
is chance of cancer
'''
#Doing prediction on a random data
random = np.random.rand(30).reshape(1,30)
print(logist_reg.predict(random))
