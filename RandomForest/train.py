from RandomForest import accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)



model = RandomForest(n_trees=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc =  accuracy(y_test, predictions)
print(f"Accuracy :{acc*100:.2f}%")