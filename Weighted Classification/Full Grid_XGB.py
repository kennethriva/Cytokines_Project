# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:20:07 2018

@author: Kenneth Rivadeneira
"""
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn import metrics

# avoid deprecation warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###LOAD BOTH TRAIN AND TEST DATASETS#####

path = '' # use path
train_df = pd.read_csv(path + "/train.csv")
test_df = pd.read_csv(path + "/test.csv")

# We will use train dataset for hyperparameter tunning and validate them
# with a split


# Drop label from X_train and X_test subsets

X_train = train_df.drop(['class'], axis=1)
y_train = train_df['class']
X_test = test_df.drop(['class'], axis=1)
y_test = test_df['class']

# GRID SEARCH CV FOR BINARY CLASSIFICATION

# To write output
import sys
old_stdout = sys.stdout

result_file = open("results.log","w")

sys.stdout = result_file


# Tunning parameters and scoring using 'roc_au'
n = 4 # number of available cores

optimized_XGB = XGBClassifier()
# change number of threads number of cores available
parameters = {'nthread':[n], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'gamma': [0.1],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [3],
              'silent': [1],
              'subsample': [0.6],
              'colsample_bytree': [0.6],
              'n_estimators': [50], #number of trees
              'scale_pos_weight': [2.7], # sum(negative cases) / sum(positive cases) = 2.7 if data is imbalanced
              'seed': [42]}

clf = GridSearchCV(optimized_XGB, parameters, n_jobs=n, 
                   cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                      random_state=42), 
                   verbose=1, refit=True,scoring='roc_auc')
clf.fit(X_train, y_train)

print("Best parameters set found on training set:")
print()
print(clf.best_params_)
print()
print("Detailed classification report:")
print()
print("The model is trained on the full training set.")
print("The scores are computed on the full test set.")
print()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print()
# probability of being classified as class 1
y_prob = clf.predict_proba(X_test)[:,1]
auroc = metrics.roc_auc_score(y_test, y_prob)
print('AUROC')
print(auroc)
print()
print('F1_score')
f1_score = metrics.f1_score(y_test, y_pred)
print(f1_score)
print()
print('Accuracy')
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print()
print('Confusion Matrix')
mat = confusion_matrix(y_test, y_pred)
print(mat)
print()
print('MC')
MC = matthews_corrcoef(y_test, y_pred)
print(MC)

sys.stdout = old_stdout

result_file.close()
