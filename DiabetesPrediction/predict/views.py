from django.db import models
from django.contrib.auth.models import User

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
# Create your views here.

def Predict(request):
    pass
def check(request):
    if request.method=='GET':
        return render(request,'check.html')

    else:
        pregnancy=request.POST.get('preg')
        gulcose=request.POST.get('glucose')
        BloodPressure=request.POST.get('bp')
        SkinThickness=request.POST.get('st')
        Insuline=request.POST.get('insulin')
        BIM=request.POST.get('bim')
        DPF=request.POST.get('dpf')
        Age=request.POST.get('age')
        mydata = [float(pregnancy or 0),float(gulcose or 120),float(BloodPressure or 112),float(SkinThickness or 12),float(Insuline or 110),float(BIM or 29),float(DPF or 0.281),float(Age or 28)]
        mydata_array = np.array(mydata)
        result= modelprepare(mydata_array);
        result = result.tolist()
        if result[0]==1:
            mypre = "Yes, you have a diabetes";
        else:
                mypre = "No you do not have diabetes";

        return render(request,'resulty.html',{'result':mypre})


def modelprepare(myarray_data):
    dataset=pd.read_csv('D:\SmartHealthPrediction System\diabetes\DiabetesPrediction\predict\diabetes_prediction.csv')
    X=dataset.iloc[:, :-1].values
    y=dataset.iloc[:, 8]
    from sklearn.preprocessing import Imputer
    imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
    imputer=imputer.fit(X[:, 0:7])
    imputer=imputer.transform(X[:, 0:7])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    myarray_data = sc_X.transform([myarray_data])
    classifier=GaussianNB()
    classifier.fit(X_train,y_train)
    result = classifier.predict(myarray_data)
    return result;
