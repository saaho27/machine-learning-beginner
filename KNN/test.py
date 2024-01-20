from KNN import KNN,accuracy
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


data = datasets.load_iris()
#using iris data set to test 
X = data.data
y = data.target

X_train , X_test ,y_train ,y_test = train_test_split(X,y ,test_size=0.25,random_state=42)


model = KNN(k=5)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(pred)

def accuracy(y_test,pred):
    accuracy = np.sum(y_test == pred) / len(y_test)
    return accuracy

acc = accuracy(y_test,pred)

print(f"Accuracy:{acc * 100 :.0f}%")
print(len(X_test),len(pred))
#make prediction for unseen data 

pre = [6.1 ,2.8 ,4.7 ,1.2]
#give the data in array
predict = model.predict([pre])
print(predict)
print(y_test[0])

