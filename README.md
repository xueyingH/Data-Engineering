# Applied Data Science in F1 Dataset
Created by Xueying Huang

## 1. Data Source
In this project, I use F1 data from 1950-2017 on the AWS S3 and finish coding in Databricks.

## 2. Research Question
The F1 dataset contains a number of features available. I utilize statistical models to explain a driver arrives in second place in 1950-2010.
Besides, I build machine learning models to predict the drivers that come in the second place between 2011-2017.

## 3. Methods
### data.py
1. Import and combine several datasets from S3.
2. Data wrangling and transformation.
3. Tackling with missing values by multiple imputing methods, including Mean, Forward Fill and KNN inputers.
4. Store the clean and complete dataset into S3 processed folder, the dataset is called complete_f1_data.csv.

### Explanatory Modeling
1. Make summary statistics and visualizations on different features in complete_f1_data.csv and compare the results of position 2 and other positions.
2. Select the meaningful features to build model.
3. Use Logit Method from statsmodels.api to regress the position on these features and summarize the regression result.
4. Use margin effect to explore more.

### Predictive Modeling
1. Split dataset, use data from 1950-2010 to train the model and predict in data from 2011-2017.
2. Addressing problems caused by imbalanced data with upsampling method.
3. Build Machine Learning Models (Random Forest, XGBoost, KNN), get metrics  and log the models.
4. Get the importance of features and make datavisualizations.


 
