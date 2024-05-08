#Package Imports
from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

#For Reproducibility
seed = 42
np.random.seed(seed)

#Data generation
n = 1000 #Number of observation
p = 50 #Number of predictors
data = pd.DataFrame(np.random.normal(size=(n, p)), columns=[f'X{i+1}' for i in range(p)])

#Induce Sparsity
sparsity_level = 0.7
data = data.mask(np.random.rand(*data.shape) < sparsity_level, 0)

data['y'] = (1
             + np.sin(data['X1']) * np.log1p(np.abs(data['X2']))
             + np.where(data['X5'] < 1, data['X3']**2 * data['X4'],0)
             + np.where(data['X3']>1, np.sqrt(np.abs(data['X5'])) * 5,0)
             + np.where(data['X6'] > 0, data['X6'], 0) ** 3) + np.random.normal(size=n) #Construct predictor variable

#Split data into train and test datasets
train_indices = np.random.choice(data.index, size=int(n * 0.8), replace=False)
DataTrain = data.loc[train_indices]
DataTest = data.drop(train_indices)


 
#### XGBoost model ####
model = XGBRegressor(random_state=seed) 

#Parameter space for XGBoost
param_space = {
    'max_depth': Integer(4, 10),
    'min_child_weight': Integer(2, 10),
    'gamma': Real(0, 0.9),
    'subsample': Real(0.2, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'lambda': Real(1.5, 10),
    'alpha': Real(0, 10),
    'learning_rate': Real(0.01, 0.2),
    'n_estimators': Integer(1000, 2000)
}

#Bayes search with cross-validation for XGBoost
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1,
                              random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
XGBoostBestModel = bayes_search.best_estimator_ #Select best model
predictions = XGBoostBestModel.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time


print("XGBoost Best parameters found: ", bayes_search.best_params_)
print("XGBoost Test MSE: ", test_mse)
print("XGBoost Average Model Training Time: ", np.mean(mean_fit_times), "seconds")


#### Gradient Boosting Model ####
model = GradientBoostingRegressor(random_state=seed)

#Parameter Space for Gradient Boosting
param_space = {
    'max_depth': Integer(1, 10),
    'min_samples_split': Integer(2, 10),
    'learning_rate': Real(0.01, 0.2),
    'n_estimators': Integer(100, 2000)
}

#Bayes search with cross-validation for Gradient Boosting
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1,
                              random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
GBbest_model = bayes_search.best_estimator_ #Select best model
predictions = GBbest_model.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time

print("Gradient Boost Best parameters found: ", bayes_search.best_params_)
print("Gradient Boost Test MSE: ", test_mse)
print("Gradient Boost Average Model Training Time: ", np.mean(mean_fit_times), "seconds")


#### Bagging Model ####
base_estimator = DecisionTreeRegressor(random_state=seed)
bagging_model = BaggingRegressor(base_estimator=base_estimator, random_state=seed)

#Parameter space for Bagging
bagging_param_space = {
    'base_estimator__max_depth': Integer(1, 15),
    'max_samples': Real(0.5, 1.0),  #Percentage of samples for each base estimator
    'n_estimators': Integer(100, 1000)  #Number of trees
}

#Bayes search with cross-validation for Bagging
bayes_search = BayesSearchCV(estimator=bagging_model, search_spaces=bagging_param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
BaggingBestModel = bayes_search.best_estimator_ #Select best model
predictions = BaggingBestModel.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time

print("Bagging Best parameters found: ", bayes_search.best_params_)
print("Bagging Test MSE: ", test_mse)
print("Bagging Average Model Training Time: ", np.mean(mean_fit_times), "seconds")

#### Ridge Model ####
model = Ridge(random_state=seed)

#Parameter space for Ridge
ridge_param_space = {
    'alpha': Real(0.01, 100)
}

#Bayes search with cross-validation for Ridge Regression
bayes_search = BayesSearchCV(estimator=model, search_spaces=ridge_param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
RidgeBestModel = bayes_search.best_estimator_ #Select best model
predictions = RidgeBestModel.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time

print("Ridge Best parameters found: ", bayes_search.best_params_)
print("Ridge Test MSE: ", test_mse)
print("Ridge Average Model Training Time: ", np.mean(mean_fit_times), "seconds")


#### SVM Regression #### 
model = SVR()

#Parameter space for SVR
svr_param_space = {
    'C': Real(0.1, 1000),
    'epsilon': Real(0.01, 1),
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'gamma': Real(0.001, 1, prior='log-uniform')  # Only applies to non-linear kernels
}

#Bayes search with cross-validation for SVR
bayes_search = BayesSearchCV(estimator=model, search_spaces=svr_param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
SVRBestModel = bayes_search.best_estimator_ #Select best model
predictions = SVRBestModel.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time

print("SVR Best parameters found: ", bayes_search.best_params_)
print("SVR Test MSE: ", test_mse)
print("SVR Average Model Training Time: ", np.mean(mean_fit_times), "seconds")


#### K-Neighbors Regression ####
model = KNeighborsRegressor()

# Parameter space for k-NN
knn_param_space = {
    'n_neighbors': Integer(1, 30),
    'weights': Categorical(['uniform', 'distance']),
    'p': Integer(1, 2)  # 1 for Manhattan distance, 2 for Euclidean distance
}


#Bayes search with cross-validation for K-Neighbors Regression
bayes_search = BayesSearchCV(estimator=model, search_spaces=knn_param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

#Model evaluation
KNNBestModel = bayes_search.best_estimator_ #Select best model
predictions = KNNBestModel.predict(DataTest.drop('y', axis=1)) #Make pred on test data
test_mse = mean_squared_error(DataTest['y'], predictions) #MSE_test
cv_results = bayes_search.cv_results_
mean_fit_times = cv_results['mean_fit_time'] #find average model train time

print("K-Neighbors Best parameters found: ", bayes_search.best_params_)
print("K-Neighbors Test MSE: ", test_mse)
print("K-Neighbors Average Model Training Time: ", np.mean(mean_fit_times), "seconds")