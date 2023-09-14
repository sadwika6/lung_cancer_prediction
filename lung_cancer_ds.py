# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:00:50 2023

@author: sadwika sabbella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\sadwika sabbella\Desktop\pyfh\survey lung cancer.csv")

print(data.head())
print(data.shape)

print(data.isna().sum())


#data["GENDER"]=data["GENDER"].map({"M":0,"F":1})# to convert string data to numbers
from sklearn.preprocessing import LabelEncoder #preprocesssing step to convert string to integer
le=LabelEncoder()
data["GENDER"]=le.fit_transform(data["GENDER"])

x=np.array(data.iloc[: , :-1])
y=np.array(data.iloc[ : ,-1])
print(x.shape,y.shape)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=2)
model.fit(xtrain, ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, ypred)*100)