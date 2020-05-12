# Databricks notebook source
import pandas as pd 
import numpy as np
import datetime as dt

# COMMAND ----------

# MAGIC %md ## 1. Combine Datasets and Data Transformation

# COMMAND ----------

# Import dataset about race results and drop the redundant columns
results = spark.read.csv('/mnt/xh2434-gr5069/raw/results.csv', header = True, inferSchema = True).toPandas()
results = results.drop(['resultId','position','positionText','time','fastestLapTime'], axis = 1)
results.head()

# COMMAND ----------

# Import dataset about stop time in pit stops
pit_stops = spark.read.csv('/mnt/xh2434-gr5069/raw/pit_stops.csv', header = True, inferSchema = True).toPandas()
pit_stops = pit_stops[['raceId', 'driverId', 'stop', 'lap', 'milliseconds']]

# COMMAND ----------

# Aggregate data and got the stop counts and average stop time of each driver in each race
stop_count = pit_stops.groupby(['raceId','driverId'])['stop'].agg('sum').to_frame()
stop_total_time = pit_stops.groupby(['raceId', 'driverId'])['milliseconds'].agg('sum').to_frame()
stop_df = stop_count.join(stop_total_time, on =['raceId', 'driverId'])
stop_df['avg_stop_time'] = stop_df.milliseconds.div(stop_df.stop).astype('int')
stop_df = stop_df.drop('milliseconds', axis = 1).rename(columns = {'stop':'stop_count'}).reset_index()
stop_df.head()

# COMMAND ----------

# Import constructor dataset and keep the information about win counts of each constructor
constructor = spark.read.csv('/mnt/xh2434-gr5069/raw/constructor_standings.csv', header = True, inferSchema = True).toPandas()
constructor = constructor[['raceId', 'constructorId', 'wins']].rename(columns ={'wins':'constructor_wins'})

# COMMAND ----------

# Import race dataset and keep the year of each race
race = spark.read.csv('/mnt/xh2434-gr5069/raw/races.csv', header = True, inferSchema = True).toPandas()
race = race[['raceId','year']]

# COMMAND ----------

# Import driver dataset to get the birth year of each driver
driver = spark.read.csv('/mnt/xh2434-gr5069/raw/drivers.csv', header = True, inferSchema = True).toPandas()
driver = driver[['driverId', 'dob', 'nationality']]

# COMMAND ----------

# combine all datasets
data = pd.merge(results, stop_df, how = 'left', on = ['raceId', 'driverId'])
data = pd.merge(data, constructor, how = 'left', on = ['raceId', 'constructorId'])
data = pd.merge(data, race, how = 'left', on = 'raceId')
data = pd.merge(data, driver, how = 'left', on = 'driverId')

# COMMAND ----------

# Calculate the age of drivers
data['age'] = data.year.sub(data.dob.dt.year)
data = data.drop('dob', axis = 1)
data.head()

# COMMAND ----------

# MAGIC %md ## 2. Cleaning Data and Dealing with Missing Values

# COMMAND ----------

# Tackling with non-numeric variable nationality, transform data type to category and use label encoding
mod_data = data.copy()
mod_data.nationality = mod_data.nationality.astype('category')
mod_data.nationality = mod_data.nationality.cat.codes
mod_data.head()

# COMMAND ----------

# Transforming all non-standard missing values to NaN
mod_data = mod_data.replace('\\N',np.NaN)

# COMMAND ----------

# Check the data type of each column
# columns "number", "milliseconds", "fastestLap", "rank", "fastestLapSpeed", "stop_count", "avg_stop_time", "constructor_wins" have missing values
mod_data.info()

# COMMAND ----------

# number column has just 6 missing values, so I delete these rows directly
mod_data = mod_data.dropna(subset = ['number'])
mod_data.number = mod_data.number.astype('int')

# COMMAND ----------

# Use KNNImputer to impute the missing values in "milliseconds", "fastestLap", "rank", "fastestLapSpeed"
from fancyimpute import KNN
imputed_lap = KNN(k=5).fit_transform(mod_data.iloc[:,0:12])

# COMMAND ----------

# replace the columns with KNN imputed result
mod_data.iloc[:,8:12] = imputed_lap[:,8:]
mod_data[["milliseconds", "fastestLap", "rank"]] = mod_data[["milliseconds", "fastestLap", "rank"]].astype('int')
mod_data.fastestLapSpeed = mod_data.fastestLapSpeed.round(3)

# COMMAND ----------

# fill all missing values in constructor_wins column with 0
mod_data.constructor_wins = mod_data.constructor_wins.fillna(0).astype('int')

# COMMAND ----------

# Use the average stop counts and stop time of each driver to impute the missing values in "stop_count", "avg_stop_time"
by_driver = mod_data.groupby('driverId')
def impute_mean(series):
    return series.fillna(series.mean())
mod_data.stop_count = by_driver["stop_count"].transform(impute_mean)
mod_data.avg_stop_time = by_driver["avg_stop_time"].transform(impute_mean)

# COMMAND ----------

# Fill the rest missing values in these two columns with forward fill method
mod_data[['stop_count', 'avg_stop_time']] = mod_data[['stop_count', 'avg_stop_time']].fillna(method = 'ffill')
mod_data.stop_count = mod_data.stop_count.round(1)
mod_data.avg_stop_time = mod_data.avg_stop_time.round(3)

# COMMAND ----------

cleaned_data = mod_data.copy()

# COMMAND ----------

# Transform "positionOrder" column, identify those arrived in second place and others, make a new categorical variable "position"
cleaned_data['position'] = cleaned_data.positionOrder.where(cleaned_data.positionOrder == 2, 0).astype('category')
cleaned_data.drop(['positionOrder'], axis = 1, inplace = True)

# COMMAND ----------

# We can see there are 1027 second place and 23587 others
cleaned_data.position.value_counts()

# COMMAND ----------

# check the data types and missing values in each column again
cleaned_data.info()

# COMMAND ----------

# Display the head of cleaned data
cleaned_data.head()

# COMMAND ----------

from io import BytesIO, StringIO
import boto3

# COMMAND ----------

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key=''
)

# COMMAND ----------

s3 = session.resource('s3')
csv_buffer = StringIO()
cleaned_data.to_csv(csv_buffer)
s3 = session.resource('s3')
s3.Object('xh2434-gr5069', 'processed/Final Project/complete_f1_data.csv').put(Body=csv_buffer.getvalue())
