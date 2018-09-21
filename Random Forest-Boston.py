# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:35:07 2018

@author: XT19143
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target

boston = datasets.load_boston()
df= pd.DataFrame(boston.data)
df.columns=boston.feature_names

# =============================================================================
# **************Pseudocode****************
# Preprocess data (Missing Values, Imputing)
# Preapare X and y
# Split the Test and Train Data
# Scale the data if needed
# Convert Categorical Values to binay if needed
# Build a model
# Fit the model
# Find the accuracy
# =============================================================================


#Preprocessing data

df.isna().sum()

#Preaparing X and y

X=df.values
y=boston.target

# Split the Test and Train data 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scaling/Normalizing the data 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# Model Preparation
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10, criterion='mse')
regressor.fit(X_train, y_train)

#Fit the model
y_pred = regressor.predict(X_test)


#Find the accuracy of Model

import statsmodels.api as sm
X = sm.add_constant(X)
model = sm.OLS(y, X_opt).fit() ## sm.OLS(output, input)
predictions = model.predict(X_opt)
model.summary()

X_opt=X[:,[0,1,2,4,5,6,8,9,10,11,12]]
model = sm.OLS(y, X_opt).fit() ## sm.OLS(output, input)
predictions = model.predict(X_opt)
model.summary()

# =============================================================================
# Conclusion: Random Forest model predicts with 74.1 accuracy. 
#             Eliminating the high P value columns gives 67.7 accuracy.
#             So All the columns are taken into consideration with ~74 % accuracy
# =============================================================================
            



















