# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:31:58 2019

@author: rivanshu
"""

import pandas as pd
import numpy as np
import seaborn as sns

col_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
            'sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df_train = pd.read_csv(r"C:\Users\rivanshu\Downloads\adult.data",names = col_names)
df_test = pd.read_csv(r"C:\Users\rivanshu\Downloads\adult.test",names = col_names)

print(df_train.shape)
print(df_test.shape)

df_test=df_test[1:]
df_test.reset_index(drop=True,inplace=True)

df_train.replace(to_replace = ' ?',value=np.nan,inplace = True)
df_test.replace(to_replace = ' ?',value=np.nan,inplace = True)
df_test.replace(to_replace = ' <=50K.',value=' <=50K',inplace = True)
df_test.replace(to_replace = ' >50K.',value=' >50K',inplace = True)
df_test.replace(to_replace = 'Private',value=' Private',inplace = True)

df_train.isnull().sum()
df_test.isnull().sum()
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
df_train.isnull().sum()
df_test.isnull().sum()

df_train[df_train.income == ' <=50K'].describe()
random_sample1 = df_train[df_train.income == ' <=50K'].sample(n = 8000,replace = False,random_state = 0)
random_sample2 = df_train[df_train.income == ' >50K'].copy()
new_train = pd.concat([random_sample1,random_sample2])
new_train = new_train.sample(frac = 1).reset_index(drop=True)
new_test = df_test.copy()

num_attributes = new_train.select_dtypes(include=['int64'])
num_attributes.hist(figsize=(10,10))
cat_attributes = new_train.select_dtypes(include=['object'])
sns.countplot(y='education', hue='income', data = cat_attributes)

labels = list(cat_attributes.columns)
from sklearn.preprocessing import LabelEncoder
for i in labels:
    le = LabelEncoder()
    le.fit(new_train[i])
    new_train[i] = le.transform(new_train[i])
    new_test[i] = le.transform(new_test[i])

X_train = new_train.drop(['income','fnlwgt'], axis =1)
Y_train = new_train['income']

X_test = new_test.drop(['income','fnlwgt'], axis =1)
Y_test = new_test['income']  

sns.heatmap(new_train.corr())  

from sklearn.ensemble import RandomForestClassifier
model_1 = RandomForestClassifier(n_estimators=100,bootstrap=True,random_state=0)
model_1.fit(X_train,Y_train)

pred_randfor = model_1.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(pred_randfor, Y_test.values)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_1, X_train, 
         Y_train, cv=5)
print(np.mean(scores)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}