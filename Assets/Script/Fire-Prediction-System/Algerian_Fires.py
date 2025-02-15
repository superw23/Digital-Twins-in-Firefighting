
"""
Part of the AI implementation in this project was adapted from an open-source Fire Prediction System 
available on GitHub, created by developer vaniseth. This code was used to establish a preliminary 
model for fire prediction, and then it was refined to suit our Digital Twin system. 
The original repository can be accessed at: https://github.com/vaniseth/Forest-Fire-Prediction-System
"""

"""
Algerian Fires Dataset Analysis for Digital Twin Fire Prediction

This Python script is dedicated to the analysis of the Algerian Fires dataset,
which is a rare and valuable compilation of fire occurrence data in Algeria.
The dataset encompasses records from two distinct regions of Algeria and is
classified into 'fire' and 'not fire' instances, which makes it particularly
useful for developing predictive models in fire management systems.

Data Set Information:
- Contains 244 instances of fire data from two regions in Algeria.
- Each region contributes 122 instances, totaling 244 instances with labels
  'fire' (138 instances) and 'not fire' (106 instances).
- Timeframe of data: June 2012 to September 2012.

Attribute Information:
Weather data observations and FWI Components such as temperature, relative humidity,
wind speed, rain, and various indices from the FWI system like FFMC, DMC, DC, ISI, and BUI.

Purpose:
The aim of this script is to use machine learning techniques to build a model that will
be deployed as part of a digital twin for fire prediction.

By leveraging the power of machine learning and the specific insights offered by this dataset,
the model will serve as a critical component in a digital twin system, enhancing the ability
to anticipate fire outbreaks and enabling proactive measures in fire-prone areas.
"""
# **Weather and Fire Index (FWI) Observations**
# 1. Temp: Max temperature in Celsius degrees (22 to 42)
# 2. RH: Relative Humidity in % (21 to 90)
# 3. Ws: Wind speed in km/h (6 to 29)

# **FWI System Components**
# 5. FFMC: Fine Fuel Moisture Code (28.6 to 92.5)
# 6. DMC: Duff Moisture Code (1.1 to 65.9)
# 7. DC: Drought Code (7 to 220.4)
# 8. ISI: Initial Spread Index (0 to 18.5)
# 9. BUI: Buildup Index (1.1 to 68)
# 10. FWI: Fire Weather Index (0 to 31.1)

# **Classification Outcome**
# 
# - Fire or not Fire


import warnings
warnings.filterwarnings("ignore")
# **Import Required Library**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import bz2,pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Importing the dataset and dropping irrelevant features for analysis
df1 = pd.read_csv('Algerian_fire_cleaned-data.csv')
# Days, months, and years are excluded from the analysis to focus on environmental factors
df2 = df1.drop(['day','month','year'], axis=1)
df2.head(10)


# Splitting dataset for regression to predict the Fire Weather Index (FWI)
X = df2.iloc[:,0:10] # Input features
y= df2['FWI'] # Target variable


# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Identifying features with high correlation to eliminate redundancy
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
corr_features

# Removing features with correlation above 0.8 threshold
X_train.drop(corr_features,axis=1, inplace=True)
X_test.drop(corr_features,axis=1, inplace=True)
X_train.shape, X_test.shape

# Scaling features to standardize the dataset
def scaler_standard(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scaler_standard(X_train, X_test)


# Training the Random Forest Regressor
Random_Forest_Regressor = RandomForestRegressor()
Random_Forest_Regressor.fit(X_train_scaled, y_train)

Random_Forest_Regressor_prediction = Random_Forest_Regressor.predict(X_test_scaled)
Random_Forest_Regressor_prediction


Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Random_Forest_Regressor_prediction})    
#Actual_predicted

meanAbErr = metrics.mean_absolute_error(y_test, Random_Forest_Regressor_prediction)
meanSqErr = metrics.mean_squared_error(y_test, Random_Forest_Regressor_prediction)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, Random_Forest_Regressor_prediction))

# Output the error metrics
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# To find coefficient of determination
r2 =  r2_score(y_test, Random_Forest_Regressor_prediction)
print("R-Square:",r2)

# Hyperparameter tuning for Random Forest Regressor
param_grid =[{'bootstrap': [True, False],
'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,120],
'max_features': ['auto', 'sqrt'],
'min_samples_leaf': [1, 3, 4],
'min_samples_split': [2, 6, 10],
'n_estimators': [5, 20, 50, 100]}]

Random_Forest_Regressor = RandomForestRegressor()
Random_rf = RandomizedSearchCV(Random_Forest_Regressor,param_grid, cv = 10, verbose=2,n_jobs = -1)
Random_rf.fit(X_train_scaled, y_train)

best_random_grid=Random_rf.best_estimator_

bestrf_pred = best_random_grid.predict(X_test_scaled)
bestrf_pred

Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': bestrf_pred})    
#Actual_predicted


# Feature selection for deployment based on importance
feature_importances = Random_rf.best_estimator_.feature_importances_
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)
importance_df

# Training final model with selected features
X_train_new = X_train.drop(['Rain', 'RH'], axis=1)
X_test_new = X_test.drop(['Rain', 'RH'], axis=1)

X_train_new_scaled, X_test_new_scaled = scaler_standard(X_train_new, X_test_new)

best_random_grid.fit(X_train_new_scaled, y_train)
bestrf_pred = best_random_grid.predict(X_test_new_scaled)
bestrf_pred


# **Classification**

# Preparing the input features (X) and the target variable (y)
X = df2.iloc[:, 0:10]
y = df2['Classes']

# Splitting the dataset into a training set and a test set with a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train.shape, X_test.shape


# Identifying and removing features with high correlation to prevent multicollinearity
corr_features = correlation(X_train, 0.8)  # Correlation threshold set at 0.8
X_train.drop(corr_features, axis=1, inplace=True)  # Dropping correlated features from training set
X_test.drop(corr_features, axis=1, inplace=True)   # Dropping correlated features from test set

# Standardizing the features to have zero mean and unit variance
X_train_scaled, X_test_scaled = scaler_standard(X_train, X_test)

# **Logistic Regression Model Training**

# Initializing the Logistic Regression model and training it on the scaled training data
Logistic_Regression  = LogisticRegression()
Logistic_Regression.fit(X_train_scaled,y_train)

print('Intercept is :',Logistic_Regression.intercept_)
print('Coefficient is :',Logistic_Regression.coef_)
print("Training Score:",Logistic_Regression.score(X_train_scaled, y_train))
print("Test Score:",Logistic_Regression.score(X_test_scaled,y_test))

Logistic_Regression_Prediction = Logistic_Regression.predict(X_test_scaled)
Logistic_Regression_Prediction

# Comparing the actual class labels with the predicted ones
Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Logistic_Regression_Prediction})    
#Actual_predicted
Score = accuracy_score(y_test,Logistic_Regression_Prediction)
Classification_Report = classification_report(y_test,Logistic_Regression_Prediction)

print("Logistic Regression")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)

# Precision: Measures the accuracy of positive predictions.
# Recall: Measures the coverage of actual positive cases.
# F1 Score: Harmonic mean of precision and recall, a balance between the two.

# **Decision Tree Classifier**
# Initialize and train the Decision Tree Classifier model
Decision_Tree_Classifier = DecisionTreeClassifier()
Decision_Tree_Classifier.fit(X_train_scaled,y_train)
Decision_Tree_Classifier_prediction = Decision_Tree_Classifier.predict(X_test_scaled)
Decision_Tree_Classifier_prediction

Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Decision_Tree_Classifier_prediction})    
#Actual_predicted
Score = accuracy_score(y_test,Decision_Tree_Classifier_prediction)
Classification_Report = classification_report(y_test,Decision_Tree_Classifier_prediction)
print("Decision Tree")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)

# **Random Forest Classifier**
# Initialize and train the Random Forest Classifier model
Random_Forest_Classifier = RandomForestClassifier()
Random_Forest_Classifier.fit(X_train_scaled,y_train)

print("Training Score:",Random_Forest_Classifier.score(X_train_scaled, y_train))
print("Test Score:",Random_Forest_Classifier.score(X_test_scaled,y_test))

Random_Forest_Classifier_prediction = Random_Forest_Classifier.predict(X_test_scaled)
Random_Forest_Classifier_prediction
Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Random_Forest_Classifier_prediction})    
#Actual_predicted

Score = accuracy_score(y_test,Random_Forest_Classifier_prediction)
Classification_Report = classification_report(y_test,Random_Forest_Classifier_prediction)
print("Random Forest")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)


# **K-Nearest Neighbors Classifier**
# Initialize and train the K-Nearest Neighbors Classifier
K_Neighbors_Classifier = KNeighborsClassifier()
K_Neighbors_Classifier.fit(X_train_scaled,y_train)
print("Training Score:",K_Neighbors_Classifier.score(X_train_scaled, y_train))
print("Test Score:",K_Neighbors_Classifier.score(X_test_scaled,y_test))

K_Neighbors_Classifier_prediction = K_Neighbors_Classifier.predict(X_test_scaled)
K_Neighbors_Classifier_prediction
Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': K_Neighbors_Classifier_prediction})    
#Actual_predicted
Score = accuracy_score(y_test,K_Neighbors_Classifier_prediction)
Classification_Report = classification_report(y_test,K_Neighbors_Classifier_prediction)
print("KNeighbors Classifier")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)

# **XGBoost Classifier**
# Initialize and train the XGBoost Classifier model
xgb = XGBClassifier()
xgb.fit(X_train_scaled,y_train)

print("Training Score:",xgb.score(X_train_scaled, y_train))
print("Test Score:",xgb.score(X_test_scaled,y_test))
xgb_predic = xgb.predict(X_test_scaled)
xgb_predic

Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': xgb_predic})    
#Actual_predicted
Score = accuracy_score(y_test, xgb_predic)
Classification_Report = classification_report(y_test, xgb_predic)
print("XGboost Classifier")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)

# **Accuracy Score Results Summary**

# The summary table below shows the accuracy scores for different classifiers 
# based on the test dataset. These results help in determining the best model 
"""
| Models                  | Accuracy score |
| ----------------------- | -------------- |
| Logistic Regression     | 91.78%         |
| Decision Tree           | 94.52%         |
| Random Forest           | 97.26%         |
| K-Neighbors Classifier  | 93.15%         |
| XGBoost Classifier      | 97.26%         |
"""


# Hyperparameter Tuning and Model Selection

# **Hyperparameter Tuning for XGBoost Classifier**
params={
 "learning_rate"    : (np.linspace(0,10, 100)) ,
 "max_depth"        : (np.linspace(1,50, 25,dtype=int)),
 "min_child_weight" : [1, 3, 5, 7],
 "gamma"            : [0.0, 0.1, 0.2 , 0.3, 0.4],
 "colsample_bytree" : [0.3, 0.4, 0.5 , 0.7]}

# Initialize the RandomizedSearchCV object to perform hyperparameter tuning
Random_xgb = RandomizedSearchCV(xgb, params, cv = 10,n_jobs = -1)
Random_xgb.fit(X_train_scaled, y_train).best_estimator_

# Select the best estimator
Best_xgb = Random_xgb.best_estimator_
Best_xgb.score(X_test_scaled,y_test)

# Predict and evaluate the best XGBoost model
Bestxgb_prediction = Best_xgb.predict(X_test_scaled)
Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Bestxgb_prediction})    
#Actual_predicted
xgb_score = accuracy_score(y_test, Bestxgb_prediction)
xgb_report = classification_report(y_test, Bestxgb_prediction)
print("Best XGBoost Classifier - Accuracy Score: {:.4f}".format(xgb_score))
print(xgb_report)

# **Hyperparameter Tuning for Random Forest Classifier**
# Define a parameter grid to search for the best parameters for Random Forest
rf_params = {
    "n_estimators": [90, 100, 115, 130],
    "criterion": ['gini', 'entropy'],
    "max_depth": range(2, 20, 1),
    "min_samples_leaf": range(1, 10, 1),
    "min_samples_split": range(2, 10, 1),
    "max_features": ['auto', 'log2']
}

# Initialize the RandomizedSearchCV object to perform hyperparameter tuning
Random_rf = RandomizedSearchCV(RandomForestClassifier(), rf_params, cv=10, n_jobs=-1)
Random_rf.fit(X_train_scaled, y_train)

# Select the best estimator
Best_rf = Random_rf.best_estimator_
Best_rf.score(X_test_scaled,y_test)

# Predict and evaluate the best Random Forest model
Bestrf_pred = Best_rf.predict(X_test_scaled)
Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': Bestrf_pred})    
#Actual_predicted

Score = accuracy_score(y_test, Bestrf_pred)
Classification_Report = classification_report(y_test,Bestrf_pred)
print("FINAL Random Forest")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)


# **Model Selection using Cross-Validation**
# apply Stratified K-Fold cross-validation to understand the consistent performance of our models across different folds.
# Stratified K-Fold maintains the proportion of the target class in each fold, leading to a more reliable cross-validation result.
skfold = StratifiedKFold(n_splits= 10,shuffle= True,random_state= 0)

# Calculate cross-validation scores for all models
# Calculate cross-validation scores for all models
cv_xgb_score = cross_val_score(Best_xgb, X, y, cv=skfold, scoring='accuracy').mean()
cv_rf_score = cross_val_score(Best_rf, X, y, cv=skfold, scoring='accuracy').mean()
cv_dt_score = cross_val_score(Decision_Tree_Classifier, X, y, cv=skfold, scoring='accuracy').mean()
cv_knn_score = cross_val_score(K_Neighbors_Classifier, X, y, cv=skfold, scoring='accuracy').mean()
cv_lg_score = cross_val_score(Logistic_Regression, X, y, cv=skfold, scoring='accuracy').mean()

# Print the mean cross-validation score for each model
print(f'Mean CV Accuracy - XGBoost: {cv_xgb_score:.4f}')
print(f'Mean CV Accuracy - Random Forest: {cv_rf_score:.4f}')
print(f'Mean CV Accuracy - Decision Tree: {cv_dt_score:.4f}')
print(f'Mean CV Accuracy - KNN: {cv_knn_score:.4f}')
print(f'Mean CV Accuracy - Logistic Regression: {cv_lg_score:.4f}')

"""
| Models                | Training Score | Test Score | Accuracy | 
|-----------------------|----------------|------------|----------|
| Logistic Regression   | 0.9706         | 0.9178     | 91.78%   | 
| Decision Tree         | 0.9752         | 0.9367     | 97.52%   | 
| Random Forest         | 1.0000         | 0.9863     | 98.63%   |
| K-Neighbors           | 0.9647         | 0.9315     | 93.15%   | 
| XGBoost               | 0.9941         | 0.9726     | 97.26%   | 
"""

# Model Deployment

# The XGBoost Classifier has demonstrated superior performance over other models.
# Hence, we will utilize this model for the final deployment phase.

# Feature Importance for Deployment

X_train_new = X_train.drop(['Rain', 'RH'], axis=1)
X_test_new = X_test.drop(['Rain', 'RH'], axis=1)

# Reducing the feature set for the deployment model
X_train_new_scaled, X_test_new_scaled = scaler_standard(X_train_new, X_test_new)

xgb_model =Random_xgb.fit(X_train_new_scaled, y_train).best_estimator_
xgb_model.score(X_test_new_scaled, y_test)

xgb_model_pred = xgb_model.predict(X_test_new_scaled)
xgb_model_pred

Actual_predicted = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': xgb_model_pred})    
#Actual_predicted

Score = accuracy_score(y_test, xgb_model_pred)
Classification_Report = classification_report(y_test, xgb_model_pred)
print("Final Model XGB")
print ("Accuracy Score value: {:.4f}".format(Score))
print (Classification_Report)

# Model Serialization for Deployment
# Compressing and saving the model to a binary file using BZ2 compression, 
import bz2,pickle
file = bz2.BZ2File('Classification.pkl','wb')
pickle.dump(best_random_grid,file)
file.close()