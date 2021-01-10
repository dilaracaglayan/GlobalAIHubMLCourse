# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

X, y= make_blobs(n_samples=2000, n_features=3)

data= pd.DataFrame(X ,columns = ["column1","column2", "column3"])

print(data.head(10))

print(data.info())

print(data.isna().sum())

print(data.describe())


y= pd.DataFrame(y, columns=["label"])
print(y["label"].value_counts())



X= pd.DataFrame(X ,columns = ["column","column", "column"])
data_concat = pd.concat([X, y], axis=1)

plt.figure(figsize=(10,8))
sns.heatmap(data_concat.corr(),annot=True,fmt='.3f',color='blue',cmap='coolwarm')
plt.show()



X_train, X_test, y_train, y_test = train_test_split(data, y ,test_size=0.3, random_state=0)


clf = DecisionTreeClassifier()
#we have to define max_depth to prevent overfitting
clf.fit(X_train,y_train)
print("Train Accuracy Decision Tree:",clf.score(X_train,y_train))
print("Test Accuracy Decision Tree:",clf.score(X_test,y_test))



xgb = XGBClassifier()
xgb.fit(X_train,y_train)
print("Train Accuracy Xgboost:",xgb.score(X_train,y_train))
print("Test Accuracy xgboost:",xgb.score(X_test,y_test))


from sklearn.model_selection import GridSearchCV

#GridSearch on Xgboost Classifier
param_dict = {
    'min_child_weight':range(1,2,6),
    'max_depth':range(3,5),   
    'learning_rate': [0.0001,0.001,0.01,0.1],
    'n_estimators': [10,50,80]}

xgb_ = GridSearchCV(xgb,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)
print("Tuned: {}".format(xgb_.best_params_)) 
print("Mean of the cv scores is {:.5f}".format(xgb_.best_score_))
print("Train Score {:.5f}".format(xgb_.score(X_train,y_train)))
print("Test Score {:.5f}".format(xgb_.score(X_test,y_test)))


#GridSearch on Decision Tree Classifier
param_dict = {
    'max_depth':range(3,5,6),
    'criterion': ["gini", "entropy"]}

clf_ = GridSearchCV(clf,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(clf_.best_params_)) 
print("Mean of the cv scores is {:.5f}".format(clf_.best_score_))
print("Train Score {:.5f}".format(clf_.score(X_train,y_train)))
print("Test Score {:.5f}".format(clf_.score(X_test,y_test)))


#Data is very small and complicated. That makes big overfitting problem.
#the one with  Xgboost is more complicated. but we doesnt see any advantages of xgb

print("*"*25)
print("Test Score {:.5f}".format(xgb_.score(X_test,y_test)))