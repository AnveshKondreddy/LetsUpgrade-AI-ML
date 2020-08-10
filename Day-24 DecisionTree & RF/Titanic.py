# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:00:34 2020

@author: rb5062
"""


import pandas as pd
from sklearn import  preprocessing
from sklearn import tree

masterData=pd.read_csv('Datasets/TitanicDataset_train.csv')

x=masterData.isnull().sum()
masterData.drop('Cabin', axis=1, inplace=True)
y=masterData.isnull().sum()
dv=mastarData.Survived

encoder = preprocessing.LabelEncoder()

masterData['gender_encoded']=encoder.fit_transform(masterData.Sex)

tree_model=tree.DecisionTreeClassifier(max_depth=...)

columnsToConsider=['gender_encoded','Age','Fare']
tree_model.fit(masterData[columnsToConsider],masterData.Survived)

with open("DTree_Age_Gender_Fare.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=columnsToConsider,out_file=f);
