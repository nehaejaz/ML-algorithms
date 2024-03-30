from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
import numpy as np

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

classifier = KNN(k=5)
classifier.fit(X_train, y_train)
predictions = classifier.predictions(X_test)

id_to_label = {index: label for index, label in enumerate(iris.target_names)}
labels = [ id_to_label[p] for p in predictions]

accuracy = np.sum(predictions==y_test)/len(y_test)
print(accuracy)


