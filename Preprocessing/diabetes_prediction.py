# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:12:30 2018

@author: Bikash
"""

print("hello world")
print("hi im bikash Saud");
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataseet
dataset=pd.read_csv('diabetes_prediction.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 8]

#take care of missing values(Also deete this becouse it is not used future)
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:, 0:7])
X[:, 0:7]=imputer.transform(X[:, 0:7])
"""
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:, 0:7]=labelencoder_X.fit_transform(X[:, 0:7])
"""
#splitting data set into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#standardisation fo values(feature Scanning)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(f"\nTrain Data Here:\n\n{X_train}")
print(f"\n test data is here:\n\n{X_test}")
X_train.all()


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
c=classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.show(cm)
#Accuricy calcu;ation
"""
#visualization
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 7].min() - 1, stop = X_set[:, 7].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('(Training set)')
plt.xlabel('No. of Preegnancy.......')
plt.ylabel('Result..........')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 8].min() - 1, stop = X_set[:, 8].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""