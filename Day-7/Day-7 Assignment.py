# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 07:56:06 2020

@author: rb5062
"""


import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency,mannwhitneyu, ttest_ind
from sklearn import preprocessing
import matplotlib.pyplot as plt

masterData= pd.read_csv('general_data.csv')
label_encoder=preprocessing.LabelBinarizer()
dataset=masterData
dataset['TotalWorkingYears']=dataset['TotalWorkingYears'].fillna(11.28)
dataset.Attrition=label_encoder.fit_transform(dataset.Attrition)
dataset=dataset.dropna()
vmasterData=masterData.dropna()
masterData['TotalWorkingYears']=masterData['TotalWorkingYears'].fillna(11.28)
masterData.dropna()

masterData.Attrition=label_encoder.fit_transform(masterData.Attrition)
masterData.head()

columns_to_consider=['Age','DistanceFromHome','Education','MonthlyIncome',
                     'NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
                     'TrainingTimesLastYear','YearsAtCompany',
                     'YearsSinceLastPromotion', 'YearsWithCurrManager']

# fig1=plt.figure()
# ax11 = fig1.add_subplot(131)
# ax12 = fig1.add_subplot(132)
# ax13 = fig1.add_subplot(133)

# ax11.boxplot(masterData.Age,labels=['Age'])
# ax12.boxplot(masterData.DistanceFromHome,labels=['DistanceFromHome'])
# ax13.boxplot(masterData.Education,labels=['Education'])

# fig2=plt.figure()
# ax21=fig2.add_subplot(131)
# ax22=fig2.add_subplot(132)
# ax23=fig2.add_subplot(133)

# ax21.boxplot(masterData.MonthlyIncome,labels=['MonthlyIncome'])
# ax22.boxplot(masterData.NumCompaniesWorked,labels=['NumCompaniesWorked'])
# ax23.boxplot(masterData.PercentSalaryHike,labels=['PercentSalaryHike'])

# fig3=plt.figure()
# ax31=fig3.add_subplot(131)
# ax32=fig3.add_subplot(132)
# ax33=fig3.add_subplot(133)
# ax31.boxplot(masterData.TotalWorkingYears,labels=['TotalWorkingYears'])
# ax32.boxplot(masterData.TrainingTimesLastYear,labels=['TrainingTimesLastYear'])
# ax33.boxplot(masterData.YearsAtCompany,labels=['YearsAtCompany'])

# fig4=plt.figure()
# ax41=fig4.add_subplot(131)
# ax42=fig4.add_subplot(132)
# ax41.boxplot(masterData.YearsSinceLastPromotion,labels=['YearsSinceLastPromotion'])
# ax42.boxplot(masterData.YearsWithCurrManager,labels=['YearsWithCurrManager'])

#plt.show()


df_Attrition_Yes = masterData.loc[masterData.Attrition == 1]
df_Attrition_No = masterData.loc[masterData.Attrition == 0]


# Method to calculate p-Value using Mann Whiteny Test
def CalculatepValueUsingMannWhitneyTest(colName):
    stat_mann,p_mann=mannwhitneyu(df_Attrition_Yes[colName], df_Attrition_No[colName])
    return round(p_mann,5)

# Method to calculate p-Value using ttest_ind
def CalculatepValueUsingttest_ind(colName):
    stat,p=ttest_ind(df_Attrition_Yes[colName],df_Attrition_No[colName])
    return round(p,5)

def CalculatepValueUsingPearsonRTest(colName):
    stat, p = pearsonr(dataset.Attrition, dataset[colName])
    return round(p,5)

analysis_results=pd.DataFrame(columns=['Column Name','p-value MannWhitney','p-value ttest_ind','p-Value PearsonR'])

# Lopping each column to find p-Value using MannWhiteny test
for i in columns_to_consider:    
    x=pd.DataFrame([[i,CalculatepValueUsingMannWhitneyTest(i),CalculatepValueUsingttest_ind(i),CalculatepValueUsingPearsonRTest(i)]],columns=['Column Name','p-value MannWhitney','p-value ttest_ind','p-Value PearsonR'])
    analysis_results=analysis_results.append(x)

#Method to draw conclusion based on p-value
def ConclusionBasedOnPValue(pValue):
    if pValue>0.05:
        return 'No impact'
    else:
        return 'Has Impact'
    
analysis_results['Conslusions_MannWhitney']=analysis_results['p-value MannWhitney'].apply((lambda row:ConclusionBasedOnPValue(row)))
analysis_results['Conslusions_ttest_ind']=analysis_results['p-value ttest_ind'].apply((lambda row:ConclusionBasedOnPValue(row)))
analysis_results['Conslusions_PearsonR']=analysis_results['p-Value PearsonR'].apply((lambda row:ConclusionBasedOnPValue(row)))

print(analysis_results)