# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:20:07 2018

@author: Neither do I
"""

import os
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

###LOAD FULL DATASET#####
os.chdir(r"C:\Users\Neither do I\Downloads") # change for original directory
data = pd.read_csv('results_normalized_filtred.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
#### SPLIT DATA INTO TRAIN AND TEST SUBSETS #####
#20% FOR TEST AND 80% FOR TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =256)

# In[GridseacrchCV] 

optimized_XGB = XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'gamma': [0.05,0.1,0.3,0.5,0.7,0.9,1.0],
              'learning_rate': [0.01,0.015,0.025,0.05,0.1], #so called `eta` value
              'max_depth': [3,5,6,7,9,12,15,17,25],
              'min_child_weight': [1,3,5,7],
              'silent': [0],
              'subsample': [0.6,0.7,0.8,0.9,1.0],
              'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
              'n_estimators': [50,100], #number of trees
              'scale_pos_weight': [2.7], # sum(negative cases) / sum(positive cases)
              'seed': [1337]}
clf = GridSearchCV(optimized_XGB, parameters, n_jobs=-1, 
                   cv=StratifiedKFold(n_splits=3, shuffle=True, 
                                      random_state=1337), 
                   verbose=2, refit=True,scoring='roc_auc')
clf.fit(X_train, y_train)
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    

# In[Test]
y_pred = clf.predict(X_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
print('Matthew coefficient')
print()
print(matthews_corrcoef(y_test, clf.predict(X_test)))
print('Matriz de confusión')
conf_mat = confusion_matrix(y_test,clf.predict(X_test))
print(conf_mat)
