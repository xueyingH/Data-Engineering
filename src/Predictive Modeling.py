# Databricks notebook source
import pandas as pd 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sn

# COMMAND ----------

dbutils.library.installPyPI("mlflow", "1.0.0")

# COMMAND ----------

# Import complete cleaned data
f1_data = spark.read.csv('/mnt/xh2434-gr5069/processed/Final Project/complete_f1_data.csv', header = True, inferSchema = True).toPandas()
f1_data.drop('_c0', axis = 1, inplace = True)
f1_data.position = f1_data.position.astype('category')

# COMMAND ----------

# subset data from 1950 - 2010
explan_df = f1_data.set_index('year').sort_index().loc[1950:2010,].reset_index()
X = explan_df.iloc[:,:-1]
y = explan_df.iloc[:,-1]

# COMMAND ----------

# predictive data from 2011- 2017
pred_df = f1_data.set_index('year').sort_index().loc[2011:2017,].reset_index()
X_pred = pred_df.iloc[:,:-1]
y_pred = pred_df.iloc[:,-1]

# COMMAND ----------

# MAGIC %md ## 1. Random Forest Classification

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# Tackling with imbalanced data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
X_pred_resampled, y_pred_resampled = ros.fit_sample(X_pred, y_pred)

# COMMAND ----------

# Train and test on data from 1950-2010 to get the best model
rfc_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=21))

rfc_param_grid = {
    'randomforestclassifier__n_estimators': [10, 100, 1000],
    'randomforestclassifier__min_samples_leaf': [ 3, 4, 5]
}
rfc_grid = GridSearchCV(rfc_pipe, rfc_param_grid).fit(X_resampled, y_resampled)

print("RANDOM FOREST (SCALED DATA)")
print("Best Parameter: {}".format(rfc_grid.best_params_))

# COMMAND ----------

# create a random forest model with parameters from GridSearchCV and fit on data from 2011-2017
best_rfc = RandomForestClassifier(min_samples_leaf=4, n_estimators=1000, random_state= 21).fit(X_resampled, y_resampled)
print("RANDOM FOREST Test set score: {:.2f}".format(best_rfc.score(X_pred_resampled, y_pred_resampled)))

# COMMAND ----------

# Get the feature importances
feature_importance = best_rfc.feature_importances_
rfc_features = pd.concat([pd.Series(X_pred.columns.get_values()), pd.Series(feature_importance)], axis=1). rename(columns = {0:'feature', 1:'importance'})
rfc_features.set_index('feature', inplace = True)
rfc_features

# COMMAND ----------

# Plot the feature importances and have a direct comparison between features
rfc_features.sort_values(by = 'importance').plot(kind = 'barh', title = "Random Forest Classification Feature Importance")

# COMMAND ----------

# Get the metrics that can evaluate the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
rfc_pred = best_rfc.predict(X_pred_resampled)
print("Accuracy Score of Random Forest Classification: {}". format(accuracy_score(y_pred_resampled, rfc_pred)))
print("ROC AUC Score of Random Forest Classification: {}".format(roc_auc_score(y_pred_resampled, rfc_pred)))
print(classification_report(y_pred_resampled, rfc_pred))

# COMMAND ----------

import mlflow.sklearn
import tempfile
# Log Random Forest Model
with mlflow.start_run(run_name = "Random Forest Classifier") as run:
  # Create model, train it, and create predictions
  # Build a Random Forest Classifier
  rfc = RandomForestClassifier(min_samples_leaf=4, n_estimators=1000, random_state= 21)
  rfc = rfc.fit(X_resampled, y_resampled)
  rfc_predictions = rfc.predict(X_pred_resampled)
  
  # Log model
  mlflow.sklearn.log_model(rfc, "random-forest-classifier")
  
  # Create metrics
  accuracy = accuracy_score(y_pred_resampled, rfc_predictions)
  roc_auc = roc_auc_score(y_pred_resampled, rfc_predictions)
  
  # Log metrics
  mlflow.log_metric("accuracy-score", accuracy)
  mlflow.log_metric("ruc-auc_score", roc_auc)
    
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md ## 2. XGBoost Classification

# COMMAND ----------

from xgboost import XGBClassifier

# COMMAND ----------

from xgboost import XGBClassifier
# define a XGBoost Classifier Model
xgb = XGBClassifier()
# fit the model on data from 1950-2010
xgb.fit(X, y)
# get predictions of data from 2011 -2017
xgb_pred = xgb.predict(X_pred)
xgb_importance = xgb.feature_importances_

# COMMAND ----------

# get the importance of each column in the dataset
xgb_features = pd.concat([pd.Series(X_pred.columns.get_values()), pd.Series(xgb_importance)], axis=1). rename(columns = {0:'feature', 1:'importance'})
xgb_features.set_index('feature', inplace = True)
xgb_features

# COMMAND ----------

# Plot the feature importances and have a direct comparison between features
xgb_features.sort_values(by = 'importance').plot(kind = 'barh', title = "XGBoost Classification Feature Importance")

# COMMAND ----------

print("Accuracy Score of XGBoost Classification: {}". format(accuracy_score(y_pred, xgb_pred)))
print("ROC AUC Score of XGBoost Classification: {}".format(roc_auc_score(y_pred, xgb_pred)))
print(classification_report(y_pred, xgb_pred))

# COMMAND ----------

import mlflow

# COMMAND ----------

# Log XGBoost Classifier Model
with mlflow.start_run(run_name = "XGBoost Classifier") as run:
  # Create model, train it, and create predictions
  # Build a XGBoost Classifier
  xgb = XGBClassifier()
  # fit the model on data from 1950-2010
  xgb.fit(X, y)
  # get predictions of data from 2011 -2017
  xgb_pred = xgb.predict(X_pred)
  
  # Log model
  mlflow.sklearn.log_model(rfc, "xgboost-classifier")
  
  # Create metrics
  accuracy = accuracy_score(y_pred, xgb_pred)
  roc_auc = roc_auc_score(y_pred, xgb_pred)
  
  # Log metrics
  mlflow.log_metric("accuracy-score", accuracy)
  mlflow.log_metric("ruc-auc_score", roc_auc)
    
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md ## 3. K-Nearest Neighbors Model

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

knn_param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid).fit(X_resampled, y_resampled)

print("KNN for REGRESSION (SCALED DATA)")
print("Best Parameter: {}".format(knn_grid.best_params_))

# COMMAND ----------

knn_pred = knn_grid.predict(X_pred_resampled)
print("Accuracy Score of KNN Classification: {}". format(accuracy_score(y_pred_resampled, knn_pred)))
print("ROC AUC Score of KNN Classification: {}".format(roc_auc_score(y_pred_resampled, knn_pred)))
print(classification_report(y_pred_resampled, knn_pred))

# COMMAND ----------

import tempfile
import os
# Log KNN Model
with mlflow.start_run(run_name = "KNN Classifier") as run:
  
  # Log model
  mlflow.sklearn.log_model(knn_grid, "knn-classifier")
  
  # Create metrics
  accuracy = accuracy_score(y_pred_resampled, knn_pred)
  roc_auc = roc_auc_score(y_pred_resampled, knn_pred)
  
  # Log metrics
  mlflow.log_metric("accuracy-score", accuracy)
  mlflow.log_metric("ruc-auc_score", roc_auc)
    
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md ## 4. Output the predication results from XGBoost and Save to Database

# COMMAND ----------

X_pred.head()

# COMMAND ----------

pred_result = pd.concat([pd.Series(y_pred), pd.Series(xgb_pred)], axis =1).rename(columns = {'position':'actual', 0: 'predict'})
xgboost_result = pd.concat([X_pred, pred_result], axis = 1)

# COMMAND ----------

xgboost_result.head()

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
xgboost = spark.createDataFrame(xgboost_result)

xgboost.write.format('jdbc').options(
  url = "jdbc:mysql://gr-5069.czibvz2sselr.us-east-1.rds.amazonaws.com/gr_5069",
  driver = 'com.mysql.jdbc.Driver',
  dbtable = "xgboost_results",
  user = 'admin',
  password = '').mode('overwrite').save()
