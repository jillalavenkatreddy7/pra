# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:19:41 2021

@author: J Venkat Reddy
"""
#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset..

dataset=pd.read_csv("heart (1).csv")

#seperating independent varibles and depenedent variables..
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

#splitting dataset into training and testing date..

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#applying PCA

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
ex_variance=pca.explained_variance_ratio_

#fitting logisticRegression to traing dataset..
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(x_train,y_train)

#predicting..
y_pred=lr.predict(x_test)

#creating confusion matrix..
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
accu=accuracy_score(y_test, y_pred)
print(accu,"***********************************")
print("predicted result:",lr.predict(np.asarray([-1.73079,-0.958751]).reshape(1,-1)))



