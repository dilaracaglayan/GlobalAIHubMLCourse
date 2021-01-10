# -*- coding: utf-8 -*-


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("winequality.csv")

print(data.head(10))

print(data.info())


print(data.isna().sum())
#nı missing value.

print(data.describe())

print(data.columns)


#visualization
plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()

plt.figure(figsize=(16,6))
sns.countplot(x="quality", data=data)
plt.xticks(np.arange(6))
plt.xlabel("Default Payent" , color="red", size=18 , alpha=0.5)
plt.show()

print(data["quality"].value_counts())


plt.figure(figsize=(14,6))
sns.distplot(data['alcohol'], kde = True, color ='blue', bins = 8) 
plt.show()
#Feature extraction

#for first question scaling process depends on our models. 
data_2 = data[(data["quality"] !=8) & (data["quality"] !=3 )]



print("duplicated samples len: ",data_2.duplicated().sum())
# There is lots of duplicated samples but we have really small scale of sample so we have to keep it.


#most related columns with quality
plt.figure(figsize=(12,8))
plt.scatter(x="volatile acidity", y="alcohol", data = data_2[data_2["quality"]==6], color ="red")
plt.scatter(x="volatile acidity", y="alcohol", data = data_2[data_2["quality"]==5], color ="green")
plt.scatter(x="volatile acidity", y="alcohol", data = data_2[data_2["quality"]==7], color ="blue")
plt.ylabel("alcohol")
plt.xlabel("volatile acidity")
plt.show()



X = data_2.iloc[:,:11]
y = data_2.iloc[:,11]


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=0)


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
print("Decision Tree Classifier")
print("Train Accuracy :",clf.score(X_train,y_train))
print("Test Accuracy ",metrics.accuracy_score(y_test,clf_pred))
print("")


xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
print("Xgboost Classifier")
print("Train Accuracy  xgb:", xgb.score(X_train,y_train))
print("Test Accuracy ",metrics.accuracy_score(y_test,xgb_pred))
print("")

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("Random Forest Classifier")
print("Train Accuracy of Random Forest C:",rfc.score(X_train,y_train))
print("Test Accuracy ",metrics.accuracy_score(rfc_pred,xgb_pred))
print("")






#GridSearch on Xgboost Classifier
param_dict = {
    'max_depth':range(3,4),
    'min_child_weight':range(1,2,6),
    'learning_rate': [0.00001,0.001,0.01,0.1],
    'n_estimators': [10,30,50,80,100]}

xgb_grid = GridSearchCV(xgb,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)
print("Mean of the cv scores is {:.5f}".format(xgb_grid.best_score_))
print("Best Parameters{}".format(xgb_grid.best_params_))



#GridSearch on Random Forest Classifier
param_dict = {
    'n_estimators': [20,30,50,80],
    'min_samples_split': [3,4,5],
    'min_samples_leaf' :[1,2]}

rfc_ = GridSearchCV(rfc,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train) 
print("Mean of the cv scores is {:.5f}".format(rfc_.best_score_))
print("Best Parameters{}".format(rfc_.best_params_))


#GridSearch on Decision Tree Classifier
param_dict = {
    'max_depth':range(3,5),
    'criterion': ["gini", "entropy"]}

clf_grid = GridSearchCV(clf,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)
print("Mean of the cv scores is {:.5f}".format(clf_grid.best_score_))
print("Best Parameters : {}".format(clf_grid.best_params_))





#in this part we run the models with best parameters which getting with GridSearch method. 
# recall precision 
print("Decision Tree Classifier")
clf = DecisionTreeClassifier(criterion ="gini", max_depth = 3)
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
print("Train Accuracy :",round(clf.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,clf_pred),3))

plot_confusion_matrix(clf ,X_test, y_test)  
plt.title("Decision Tree Classifier")
plt.grid(False)
plt.show()
print("Decision Tree Classifier ")
print("Precision :{}".format(precision_score(y_test, clf_pred, average='macro')))
print("Recall  : {}".format(recall_score(y_test, clf_pred, average='macro')))
print("f1-score :{}".format(f1_score(y_test, clf_pred, average="macro")))
print("Accuracy  :{}".format(accuracy_score(y_test, clf_pred)))
print("****************************")


print("*************Xgboost Classifier************")
xgb = XGBClassifier(learning_rate = 0.1, max_depth= 3, min_child_weight= 1, n_estimators=100)
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
print("Train Accuracy of xgb:",round(xgb.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,xgb_pred),3))


plot_confusion_matrix(xgb ,X_test, y_test)  
plt.title("XGboost Classifier")
plt.grid(False)
plt.show()
print("XGBoost Classifier")
print("Precision : {}".format(precision_score(y_test, xgb_pred, average='macro')))
print("Recall  :{}".format(recall_score(y_test, xgb_pred, average='macro')))
print("f1-score :{}".format(f1_score(y_test, xgb_pred, average="macro")))
print("Accuracy   : {}".format(accuracy_score(y_test, xgb_pred)))
print("****************************")


print("*************Random Forest Classifier************")
rfc= RandomForestClassifier(min_samples_split= 2, n_estimators = 20)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("Train Accuracy of Random Forest C.:",rfc.score(X_train,y_train))
print("Test Accuracy",metrics.accuracy_score(y_test,rfc_pred))

plot_confusion_matrix(rfc ,X_test, y_test)  
plt.title("Random Forest Classifier")
plt.grid(False)
plt.show()

print("Random Forest Classifier")
print("Precision  : {}".format(precision_score(y_test, rfc_pred, average='macro')))
print("Recall  : {}".format(recall_score(y_test, rfc_pred, average='macro')))
print("f1-score :{}".format(f1_score(y_test, rfc_pred, average="macro")))
print("Accuracy  : {}".format(accuracy_score(y_test, rfc_pred)))
print("****************************")
#%% comments and final decision.

# Random Forest Classsifier is best option for this project.There will be a overfitting problem but since we have small data it doesnt make so much trouble.
#these tree algorithm creating to handle huge and complicated datasets. 



("Best model is Xgboost Classifier")
print("Accuracy : {}".format(accuracy_score(y_test, xgb_pred)))
print("Precision  : {}".format(precision_score(y_test, xgb_pred, average='macro')))