25# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:43:04 2020

@author: rb5062
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
masterData=pd.read_csv('Datasets/attritionDataset.csv');

columns=masterData.columns

columns_to_consider=['Age','Gender','MonthlyIncome','StockOptionLevel','TotalWorkingYears','YearsSinceLastPromotion','JobLevel','Department','EducationField','JobRole','PercentSalaryHike']

encoder = LabelEncoder()

masterData['Gender']= encoder.fit_transform(masterData.Gender)

# rf_model=RandomForestClassifier(n_estimators=1000,oob_score=True,min_samples_split=2)

# rf_model.fit(masterData[columns_to_consider],masterData['Personal Loan'])
