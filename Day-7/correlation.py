import pandas as pd
from scipy.stats import pearsonr
masterData= pd.read_csv('general_data.csv')

df=masterData.dropna()
df1=pd.DataFrame()
columns_to_consider=['Attrition_1','Age','DistanceFromHome','Education','MonthlyIncome',
'NumCompaniesWorked', 'PercentSalaryHike','TotalWorkingYears', 'TrainingTimesLastYear',
'YearsAtCompany','YearsSinceLastPromotion', 'YearsWithCurrManager']
x=df['Attrition']
df1['Attrition_1']=x
df['Attrition_1']=df['Attrition'].map({'Yes':1,'No':0})
print(df.Attrition_1.head())
x, y=pearsonr(df.Attrition_1, df.Age)
print(x, y)

#print(df[columns_to_consider].corr())
corr_df=df[columns_to_consider].corr()


def applyColorCode(val):
    color = 'red' if  val >0.1 else 'black'    
    return 'color: %s' % color

#corr_df.style.applymap(applyColorCode)
print(corr_df)