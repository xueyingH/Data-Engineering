# Databricks notebook source
import pandas as pd 
import numpy as np
import datetime as dt

# COMMAND ----------

# Import complete cleaned data
f1_data = spark.read.csv('/mnt/xh2434-gr5069/processed/Final Project/complete_f1_data.csv', header = True, inferSchema = True).toPandas()

# COMMAND ----------

# Make a little change to the imported data
f1_data.drop('_c0', axis = 1, inplace = True)
f1_data.position = f1_data.position.astype('category')

# COMMAND ----------

# Check the information of the dataset
f1_data.info()

# COMMAND ----------

# MAGIC %md ## 1. Choose Features that Make Theoretical Sense

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# set "year" as index and subset F1 racing between 1950 and 2010
explan_df = f1_data.set_index('year').sort_index().loc[1950:2010,].reset_index()
len(explan_df)

# COMMAND ----------

explan_df.head()
explan_df.tail()

# COMMAND ----------

# Get distribution about "position"
explan_df.position.value_counts()

# COMMAND ----------

# Plot boxplots for features selected
sns.boxplot(x='position', y='points', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='rank', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='constructorId', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='fastestLapSpeed', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='milliseconds', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='statusId', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='constructor_wins', data=explan_df)

# COMMAND ----------

sns.boxplot(x='position', y='age', data=explan_df)

# COMMAND ----------

# MAGIC %md ### Answer:
# MAGIC 
# MAGIC After simple exploratory data analysis, I choose several features that have obvious differences in second place and other places. 
# MAGIC 1. points: points of first place is the highest, and the points of second place is much higher than drivers at other positions
# MAGIC 2. milliseconds: This feature describes the time driver use to finish the race. The less the milliseconds, the higher chance to arrive at first or seocnd
# MAGIC 3. rank: This variable is a rank of time to finish the fastest lap.
# MAGIC 4. constructorId: The type of constructor installed on the race car might be a factor affected the race result.
# MAGIC 5. fastestLapSpeed: The fastestLapSpeed is an indicator that drivers' ability to increase speed.
# MAGIC 6. statusId: This variable represents the status of driver in the race, like if finish the race or if there are some accidents.
# MAGIC 7. constructor_wins: The number of wins of each kind of constructor used in the race car.
# MAGIC 8. age: I suppose the result is related to age. But I am not sure if younger drivers are more brave to win the race or older drivers have more experience to win.

# COMMAND ----------

# MAGIC %md ## 2. Modeling

# COMMAND ----------

# MAGIC %md ### 1) Logistic Regression Feature Importance

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

X = explan_df[['points', 'milliseconds','rank', 'constructorId', 'fastestLapSpeed', 'statusId', 'constructor_wins', 'age']]
y = explan_df[['position']]

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# COMMAND ----------

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=1)

# COMMAND ----------

# Create Logistic Model to fit data
logreg_pipe = make_pipeline(StandardScaler(), LogisticRegression())

logreg_param_grid = {'logisticregression__C': [0.1, 1, 10,100]}
logreg_grid = GridSearchCV(logreg_pipe, logreg_param_grid).fit(X_train, y_train)

print("LOGISTIC REGRESSION")
print("Best Parameter: {}".format(logreg_grid.best_params_))
print("Test set Score: {:.2f}".format(logreg_grid.score(X_test, y_test)))
# Cross Validation
print("Best cross-validation score: {:.2f}".format(logreg_grid.best_score_))

# COMMAND ----------

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

best_logreg = LogisticRegression(C = 10).fit(X_train_scaled, y_train)
coefs = pd.Series(best_logreg.coef_[0])
feature_names = pd.Series(X.columns.get_values())
pd.concat([feature_names, coefs], axis = 1).rename(columns = {0:'features', 1:'feature importance'}).set_index('features').sort_values('feature importance').plot(kind = 'bar')

# COMMAND ----------

# MAGIC %md ### 2) Logit Explanatory Model (statsmodels)

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

logit_data = explan_df.copy()
logit_data.position = logit_data.position.where(logit_data.position == 0, 1).astype('category')

# COMMAND ----------

logit = sm.Logit(logit_data[['position']], logit_data[['points', 'milliseconds','rank', 'constructorId', 'fastestLapSpeed', 'statusId', 'constructor_wins', 'age']])

# fit the model
result = logit.fit()

# COMMAND ----------

print(result.summary())

# COMMAND ----------

# MAGIC %md ### 3) Marginal Effects

# COMMAND ----------

# provide some marginal effects
print(result.get_margeff(at = 'overall').summary())
