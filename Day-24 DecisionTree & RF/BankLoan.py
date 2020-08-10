# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:16:24 2020

@author: rb5062
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

masterData=pd.read_excel('Datasets\Bank_Personal_Loan_Modelling.xlsx',sheet_name='Data');

columns=masterData.columns

columns_to_consider=['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']

rf_model=RandomForestClassifier(n_estimators=1000,oob_score=True,min_samples_split=2)

rf_model.fit(masterData[columns_to_consider],masterData['Personal Loan'])

print("OOB Score : ",rf_model.oob_score_)

for feature,imp in zip(columns_to_consider,rf_model.feature_importances_):
    print(feature, "-",imp*100)
# Income has Higest importance - 34.611
# Education - 20
# CCAvg - 16.37
# Family - 11.68
# Rest other columns has importance score of lessthan 10.
    
#Considering features with importance score > 10 for Decision Tree

d_tree_columns =['Income','Education','CCAvg','Family']

from sklearn import tree

dt_model=tree.DecisionTreeClassifier()

dt_model.fit(X=masterData[d_tree_columns],y=masterData['Personal Loan'])

with open('personalLoan.dot', 'w') as f:
    f=tree.export_graphviz(dt_model,out_file=f,feature_names=d_tree_columns)
