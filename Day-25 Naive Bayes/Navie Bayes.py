# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:50:11 2020

@author: rb5062
"""


import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

masterData=pd.read_csv('Datasets/TitanicDataset_train.csv')

hasNulls=masterData.isnull().sum()
encoder=LabelEncoder()
masterData.Sex=encoder.fit_transform(masterData['Sex'])
masterData.Embarked=encoder.fit_transform(masterData['Embarked'])
# y=masterData.Survived
# X=masterData.drop(['Cabin','PassengerId','Survived','Name','Ticket'],axis=1,inplace=False)

#dataset after removing irrelevent columns
dataSet=masterData.drop(['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=False)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# gnb=GaussianNB()

# y_pred=gnb.fit(X_train,y_train).predict(X_test)

# accuracyScore=accuracy_score(y_test,y_pred)
# confusionMatrix= confusion_matrix(y_test,y_pred)


def GetAccuracyScoreUsingGaussianNB(independentV,dependentV):
    iv=masterData[independentV]
    dv=masterData[dependentV]
    
    X_train,X_test,y_train,y_test = train_test_split(iv,dv,test_size=0.3)
    gnb=GaussianNB()
    y_pred=gnb.fit(X_train,y_train).predict(X_test)
    return accuracy_score(y_test,y_pred)
    
    
columnsAvailable=dataSet.columns

print("Survived : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('Survived'), 'Survived')*100)
print("Pclass : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('Pclass'), 'Pclass')*100)
print("Sex : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('Sex'), 'Sex')*100)
print("SibSp : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('SibSp'), 'SibSp')*100)
print("Parch : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('Parch'), 'Parch')*100)
print("Embarked : ",GetAccuracyScoreUsingGaussianNB(columnsAvailable.drop('Embarked'), 'Embarked')*100)