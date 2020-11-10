import pandas as pd
from sklearn.model_selection import train_test_split


data=pd.read_csv('mydata.csv')
data.head(10)
data.shape
data.count()

y=data.bphi
x=data.drop('bphi',axis=1)

m=x.shape[0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression as lm
model=lm().fit(x_train,y_train)

test = x_test.head(1)
predictions = model.predict(x_test.head(1))
import matplotlib.pyplot as plt
plt.scatter(y_test.head(1),predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


predictions


predictions[0:1000]


print ("Score:", model.score(x_test, y_test))
print (predictions)





