25# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:43:04 2020

@author: rb5062
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
masterData=pd.read_csv('Datasets/attritionDataset.csv');

columns=masterData.columns

columns_to_consider=['Age','Gender','MonthlyIncome','StockOptionLevel','TotalWorkingYears','YearsSinceLastPromotion','JobLevel','Department','EducationField','MaritalStatus','JobRole','PercentSalaryHike']

encoder = LabelEncoder()

masterData['Gender']= encoder.fit_transform(masterData.Gender)
masterData['Attrition']= encoder.fit_transform(masterData.Attrition)
masterData['BusinessTravel']= encoder.fit_transform(masterData.BusinessTravel)
masterData['Department']= encoder.fit_transform(masterData.Department)
masterData['EducationField']= encoder.fit_transform(masterData.EducationField)
masterData['JobRole']= encoder.fit_transform(masterData.JobRole)
masterData['MaritalStatus']= encoder.fit_transform(masterData.MaritalStatus)

masterData=masterData.dropna()
masterData=masterData.drop_duplicates()

rf_model=RandomForestClassifier(n_estimators=1000,oob_score=True,min_samples_split=2)

rf_model.fit(masterData[columns_to_consider],masterData['Attrition'])

print("oob_score : ",rf_model.oob_score)

for feature, imp in zip(columns_to_consider,rf_model.feature_importances_):
    print(feature, "  :  ",imp*100)

###### TOP influenctial features ##############
#MonthlyIncome - 0.17611
#TotalWorkingYears - 0.1477
#Age - 0.1437
#PercentSalaryHike - 10.42    
    
imp_columns=['MonthlyIncome','TotalWorkingYears','Age','PercentSalaryHike']

dt_model=tree.DecisionTreeClassifier()

dt_model.fit(X=masterData[imp_columns],y=masterData['Attrition'])

with open(file='AttritionDecisionTree.dot',mode='w') as f:
    f=tree.export_graphviz(dt_model,feature_names=imp_columns,out_file=f)