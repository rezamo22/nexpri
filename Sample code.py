# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:14:19 2021

@author: Reza
"""


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt



os.chdir(r"C:\Users\Reza\Desktop\Delta Interview")


credit = pd.read_excel("credit.xls")

header = credit.iloc[0]
credit = credit[1:]
credit.columns = header
#%%
# change to list comprehension 000000000000000000000000000000000000

for i in list(credit.columns)[6:len(credit.columns)-1]:
    # for some of them FLOAT  also could work
    credit[i] = credit[i].astype(int)
    

# check for nulls 
is_null = credit.isnull().sum()

credit["AGE"] = credit["AGE"].astype(int)
credit["LIMIT_BAL"] = credit["LIMIT_BAL"].astype(float)
#%%
# check for imbalance ratio
default_ratio = sum(credit["default payment next month"])/len(credit)

#%%

# check for outliers 
# for categorical variables check if the values match data description

for i in ["SEX", "EDUCATION", "MARRIAGE"]:
    
    print(credit[i].unique())
    
# sex is fine only two levels 1:male and 2 female 

#%%

# Education; 1 = graduate school; 2 = university; 3 = high school; 4 = others
# there are  0,5,6 as well

# since not big portion of data may delete this
ratio_educate_056 = len(credit[credit["EDUCATION"].isin([0,5,6])])/len(credit)
ratio_marriage_056 = len(credit[credit["EDUCATION"].isin([0,5,6])])/len(credit)

#credit.EDUCATION.value_counts()
credit = credit[credit["EDUCATION"].isin([1,2,3,4])]

# marriage: (1 = married; 2 = single; 3 = others).
credit.MARRIAGE.value_counts()
credit = credit[credit["MARRIAGE"].isin([1,2,3])]

#%%
# map the values and change to categorical 
sex_dict = {1: "Male", 2:"Female"}
education_dict = {1: "Graduate School", 2: "University", 3: "High School", 4:"others"}
marriage_dict = {1: "Married", 2: "Single", 3:"others"}

credit = credit.replace({"SEX": sex_dict, "EDUCATION": education_dict, "MARRIAGE":marriage_dict})

#%%
sns.distplot(credit['AGE'],kde=True,bins="auto", color = "green")

# or max min
max(list(credit["AGE"]))
min(list(credit["AGE"]))
#%%
#plt.boxplot(list(credit['LIMIT_BAL']))

sns.distplot(credit['LIMIT_BAL'],kde=True,bins=30, color = "green")

# will not see this as outliers

# there are outliers according to the plot and describe value
# lets detect outliers
# based on the plot the data is not normally distributed so NOT zscore method

# lets use IQR and see how many obs is detected as outlier
#%%
def outlier_bounds(column):
 sorted(column)
 Q1,Q3 = np.percentile(column , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range

lb, up = outlier_bounds(credit['LIMIT_BAL'])

credit = credit[(credit['LIMIT_BAL']>lb) & (credit['LIMIT_BAL']<up) ]
sns.distplot(credit['LIMIT_BAL'],kde=True,bins=30, color = "green")

# check how many records deleted
#%%
credit[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe(include='all')

credit.hist(column = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'] )


#%%
credit.boxplot(column = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'] )

bill_amt_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
# checked this removes about 20% of the data so this method is not reliable for these cols
# removing some outliers to reduce unneccessary variability in the data

# --- since these values do not look to be error remove some of them for variablility reason
for i in bill_amt_cols:

    credit = credit[(credit[i]>-250000) & (credit[i]<750000)]
#%%
#__________________________ Feature engineering ______________________________


# 1-  ratio of bill to amount of credit (remaining) for the LAST month only

credit["bill_credit_bal_ratio"] = credit['BILL_AMT1']/credit['LIMIT_BAL']
credit["bill_credit_bal_ratio"] = credit["bill_credit_bal_ratio"].astype(float)


# 2- total amount of pay in six month
credit["total_pay"] = credit.iloc[:,18:24 ].sum(axis=1)

# 3- total nbr of on-time or not delayed payments 
credit 
credit["ontime_pays"] = 0
credit["late_pays"] = 0 

for i in range(len(credit)):
    
    pays_hist = credit.iloc[i, 6:12]
    
    credit.iloc[i,len(list(credit))-2] = len(pays_hist[pays_hist<0])
    credit.iloc[i,len(list(credit))-1] = len(pays_hist[pays_hist>=0])
    


# 4- what percent of the bill in six month is paid
#%%
credit["pay_to_bill_1"] =  credit['BILL_AMT1'] - credit['PAY_0']

credit["pay_to_bill_2"] = credit['BILL_AMT2'] - credit['PAY_2']
credit["pay_to_bill_3"] = credit['BILL_AMT3'] - credit['PAY_3'] 
credit["pay_to_bill_4"] = credit['BILL_AMT4'] - credit['PAY_4']
credit["pay_to_bill_5"] = credit['BILL_AMT5'] - credit['PAY_5']
credit["pay_to_bill_6"] = credit['BILL_AMT6'] - credit['PAY_6']


#%%

# ============================  EDA   =============================

# age, sex , education , ... on each class 



sns.catplot(x='default payment next month', y='LIMIT_BAL', kind="box", hue = "SEX", data=credit)


#%%
sns.catplot(x='default payment next month', y='PAY_0', kind="swarm", data=credit)

#%%
sns.distplot(credit.loc[credit['default payment next month']==1, "PAY_0"], hist = True, kde = False)




# simple bar chart to show for each category howmany is 1 and 0

# some complicated plots as well
# some for generated features 
#%%
from statsmodels.graphics.mosaicplot import mosaic
#%%
plt.rcParams['font.size'] = 16.0
mosaic(credit, ['default payment next month','MARRIAGE'])

mosaic(credit, ['default payment next month','SEX'])

#%%
mosaic(credit, ['EDUCATION','default payment next month'])

plt.xticks(rotation = 90)

#%%



#%%
# 5- create the trend estimate with linear regression if you can

# colinearity and multicolinearity 

sns.heatmap(credit.corr(), xticklabels=True, yticklabels=True, annot = False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.savefig("corr.png", dpi =600)

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
cols = list(credit)
cols.remove('default payment next month')
vif_data["feature"] = cols[1:]
#%%
#------- error becasue object type has to be changed to others
vif_data["VIF"] = [variance_inflation_factor(credit.values, i)
                          for i in range(6,len(vif_data))]


#%%

credit_dum = pd.get_dummies(credit, prefix=["SEX", "EDUCATION","MARRIAGE" ] , columns=["SEX", "EDUCATION","MARRIAGE" ]  )
credit_dum = credit_dum.drop(["ID"], axis =1 )

#%%
# feature selection
#___ since interpretability is important here..
# instead of PCA we use feature selection

# X and y
x_credit_dum = credit_dum.drop(['default payment next month'], axis =1 )
#%%
y_credit_dum = credit_dum['default payment next month'].astype('category')

#%%
# stratified test and train(will go for cross validation)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_credit_dum, y_credit_dum,\
                                                    random_state=0, stratify= y_credit_dum, test_size=0.2, shuffle=True)
                                                    
#%%%

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
from sklearn import feature_selection
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
#%%

# tried recursive feature elimination took long and stopped
pipeline = imbpipeline(steps = [
                                ['scaler', MinMaxScaler()],
                                ['smote', SMOTE(random_state=11)],
                                ['selector', SelectKBest(mutual_info_classif, k=16)],
                                ['classifier', xgb.XGBClassifier()]])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

search_space = [{'selector__k': [10]},
                {'classifier': [xgb.XGBClassifier()],
                 'classifier__max_depth': [2,5],
                 'classifier__eta': [0.1],
                 'classifier__subsample': [0.6],
                 'classifier__n_estimators': [200]},
                {'classifier': [RandomForestClassifier()],
                 'classifier__max_depth': [2, 5],
                 'classifier__n_estimators': [200],
                 'classifier__max_features': [8]}]
                 

#%%
clf = GridSearchCV(pipeline, search_space, scoring='f1', cv= stratified_kfold, verbose=2)
clf = clf.fit(X_train, y_train)
#%%
# write the results to a dataframe 
results = pd.DataFrame(clf.cv_results_)

clf.best_estimator_

clf.best_score_


#%%
# check for overfitting ???


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


y_pred = clf.predict(X_test)

print('f1_score is: %.3f' % f1_score(y_test, y_pred))
print('Recall is: %.3f' % recall_score(y_test, y_pred))
print('Precision is: %.3f' % precision_score(y_test, y_pred))


#%%
# print ROC curve


#print confusion matrix 
test = y_test
pred = y_pred

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test, pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

#print (roc_auc_score(test, pred))
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print('ROC AUC (area under curve) %.3f' % roc_auc)
#%%
# precision - recall plot

from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
# plot the model precision-recall curve

plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision  - Recall curve")


#%%

#%%
pipeline.named_steps['xgboost']

pipeline.steps[1]
#%%
clf.best_estimator_._final_estimator.feature_importances_

#%%
support = pipeline.named_steps['selector'].get_support()

pipeline.get_params()
#%%
skb_step = clf.best_estimator_.named_steps['selector']


features_index = skb_step.get_support(indices=True)

selected_features = []
for i in features_index:
    
    selected_features.append(list(X_train)[i])
    
    
feature_score = clf.best_estimator_._final_estimator.feature_importances_

#%%

plt.bar(selected_features, feature_score, width=0.5)
plt.xlabel("Feature name")
plt.xticks(rotation = 90)
plt.ylabel("importance score")

#%%
results = pd.DataFrame(clf.cv_results_)

#%%
from sklearn.inspection import plot_partial_dependence

plot_partial_dependence(clf, credit, features = [2]) 

#%%


