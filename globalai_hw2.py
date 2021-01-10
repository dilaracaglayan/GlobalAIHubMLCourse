# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_boston
from scipy import stats


X, y =  load_boston(return_X_y = True)


df_boston = pd.DataFrame(X ,columns = load_boston().feature_names)
print(df_boston.head(10))
print(df_boston.describe())
print(df_boston.columns)
print(df_boston.isna().sum())
print(df_boston.info())



plt.figure(figsize=(18,10))
sns.heatmap(df_boston.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()



z = np.abs(stats.zscore(df_boston))

outliers = list(set(np.where(z > 3)[0]))
new_df = df_boston.drop(outliers,axis = 0).reset_index(drop = False)

print("outlier samp. :{}".format(z))
print("number of outliers: {}".format(len(outliers)))


X_new = new_df.drop('index', axis = 1)
y_new = y[list(new_df["index"])]
print(len(y_new))
print(len(X_new))


X_train, X_test, y_train, y_test = train_test_split(X_new, y_new ,test_size=0.3, random_state=0)
print(X_train.shape)



from sklearn.linear_model import  LinearRegression, Ridge, Lasso
regression = LinearRegression()
model = regression.fit(X_train, y_train)
print("Linear Regression Train: ", regression.score(X_train, y_train))
print("Linear Regression  Test: ", regression.score(X_test, y_test))
print('intercept:', model.intercept_)

importance = model.coef_
for i in range(len(importance)):
    print("Feature", df_boston.columns[i], "Score:", importance[i])

print('************')


alpha_values= [1, 0.1, 0.01, 0.001, 0.0001]


print("********Ridge Regression *********")

for i in alpha_values:
    ridge_model = Ridge(alpha = i)
    ridge_model.fit(X_train, y_train)
    print("Ridge Train: ", ridge_model.score(X_train, y_train))
    print("Ridge Test: ", ridge_model.score(X_test, y_test))
    print('**************')



print("***********Lasso Regression *************")
for i in alpha_values:
    lasso_model = Lasso(alpha = i)
    lasso_model.fit(X_train, y_train)
    print("Lasso Train: ", lasso_model.score(X_train, y_train))
    print("Lasso Test: ", lasso_model.score(X_test, y_test))
    print('**************')







#best alpha value is 0.01 for lasso regresion models 
#best alpha value is 0.1 for ridge regresion models 


print("best performing model is;")
ridge_model = Ridge(alpha = 0.1)
ridge_model.fit(X_train, y_train)
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))







