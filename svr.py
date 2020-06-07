import numpy as np
import matplotlib as plt
import pandas as pd

dataset=pd.read_csv('Cars_Purchasing_Data.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

y.reshape(len(y),1)

from sklearn.preprocessing import StandardScalar
sc_X=StandardScalar()
sc_y=StandardScalar()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)

y_test=sc_y.inverse_transform(y_test)
y_pred=sc_y.inverse_transform(regressor.predict(X_test))

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))