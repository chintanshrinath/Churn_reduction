# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:53:01 2018

@author: Chintan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# =============================================================================
# Importing the dataset
# =============================================================================
train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')

# =============================================================================
# Describing the data
# =============================================================================
train.describe()
print(train.dtypes)

# =============================================================================
# Checking if dataset has null values
# =============================================================================
print(train.info())

# =============================================================================
# Information on churn# 
# =============================================================================
y=train["Churn"].value_counts()
import seaborn as sns
sns.barplot(y.index, y.values)
# =============================================================================
# Descriptive Analysis
# =============================================================================
df=pd.DataFrame(np.random.randn(10,5),
                columns=['total day calls','total day minutes',
                         'internation plan','voice mail plan','Churn'])
boxplot = df.boxplot(column=['total day calls','total day minutes',
                             'internation plan','voice mail plan','Churn'])
# =============================================================================
# Churn By Internation plan 
# =============================================================================
train.groupby(["international plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
train.groupby(["voice mail plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
train.groupby(["area code", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
train.groupby(["number customer service calls", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

# ===========================================================================
# Churn By State
# =============================================================================

train.groupby(["state","international plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10)) 

# =============================================================================
# Dropping unwanted columns from train & testtdataset
# =============================================================================
train=train.drop(['state','area code','phone number','international plan',
                  'voice mail plan'],axis=1)
test=test.drop(['state','area code','phone number','international plan',
                'voice mail plan'],axis=1)

# =============================================================================
# Splitting the dataset into Train and 
#Test data according to the dimensions needed
# =============================================================================
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 15].values
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,15].values
# =============================================================================
# Encoding the Dependent Variable
# =============================================================================
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.fit_transform(y_test)
#==============================================================================
# Feature ranking with recursive feature elimination
#==============================================================================
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model=LogisticRegression()
rfe=RFE(model,2)
fit = rfe.fit(X_train,y_train)
n_features=fit.n_features_
support=fit.support_
ranking=fit.ranking_

X_train=X_train[:,[12,14]]
X_test=X_test[:,[12,14]]

# =============================================================================
# Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# Fitting classifier to Random Forest Classifer training set
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, 
                                    criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# =============================================================================
# preidicting test set results
# =============================================================================
y_pred=classifier.predict(X_test)
print(y_pred)

# =============================================================================
# Confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn import metrics
print('Random Foreset Score =',round(metrics.accuracy_score(y_test, y_pred),2))

conf = (metrics.confusion_matrix(y_test, y_pred))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# =============================================================================
# Fitting classifier to Naive bayes training set
# =============================================================================
from sklearn.naive_bayes import GaussianNB
classifier1=GaussianNB()
classifier1.fit(X_train,y_train)
# =============================================================================
# preidicting test set results
# =============================================================================
y_pred_n=classifier1.predict(X_test)

# =============================================================================
# Confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm_n=confusion_matrix(y_test,y_pred_n)

from sklearn import metrics
print('Naive Bayes Score =',round(metrics.accuracy_score(y_test, y_pred_n),2))

conf = (metrics.confusion_matrix(y_test, y_pred_n))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
