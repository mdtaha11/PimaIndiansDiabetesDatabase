# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:30:35 2020

@author: Taha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("diabetes.csv")

corr=data.corr()
sns.heatmap(corr,annot=True)

from sklearn.neighbors import LocalOutlierFactor
clf=LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(data)
 
scores = clf.negative_outlier_factor_
(y_pred==-1).sum()
data=data.drop(data[y_pred==-1].index,axis=0)

#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#data=sc.fit_transform(data)

data.isnull().sum()

data['Insulin'].value_counts()
data['SkinThickness'].value_counts()
data['BloodPressure'].value_counts()

data.hist()

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer =SimpleImputer(missing_values=0,strategy='median')
X_train2 = pd.DataFrame(imputer.fit_transform(X_train))
X_test2 = imputer.transform(X_test)

sns.FacetGrid(data,hue='Outcome',height=5).map(plt.scatter,"SkinThickness","Insulin").add_legend()

data.Age.value_counts()
plt.hist(data['Age'])
plt.hist(data[data['Outcome']==1]['Age'])
data[data['Outcome']==1]['Age'].value_counts().sum()

def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)
    plotOne.map(sns.distplot,feature,kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()

plt.plot(data['Outcome'],data['Age'])
plt.hist(data['Outcome'])
plotHistogram(data,"Outcome","Age",'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train2=pd.DataFrame(sc.fit_transform(X_train2))
sc=StandardScaler()
X_test2=sc.fit_transform(X_test2)


sns.boxplot(x=X_train2[4])

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [1.0, 10.0, 50.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
}

model_svc = SVC()

grid_search = GridSearchCV(
    model_svc, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train2, y_train)


classifier=SVC()
classifier.fit(X_train2,y_train)
y_pred=classifier.predict(X_test2)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

