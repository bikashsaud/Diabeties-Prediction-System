# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 06:32:14 2018

@author: Bikash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing dataseet
dataset=pd.read_csv('diabetes_prediction.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 8]

#splitting data set into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#standardisation fo values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


