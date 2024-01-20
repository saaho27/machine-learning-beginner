from sklearn import datasets
from svm import SVM
from svm import accuracy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import numpy as np
X , y = datasets.make_blobs(
    n_samples=50,n_features=2 ,centers= 2 , cluster_std=1.05 , random_state= 42
)

y  = np.where( y == 0,-1,1)
 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=123)

model = SVM()
model.fit(X_train , y_train)

pred = model.predict(X_test)

acc = accuracy(y_test,pred)

print(f"Accuracy:{acc * 100 :.0f}%")


#visualization 

