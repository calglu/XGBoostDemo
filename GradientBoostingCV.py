from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Seed for reproducibility
seed = 42
np.random.seed(seed)

# Data generation
n = 1000
p = 50
data = pd.DataFrame(np.random.normal(size=(n, p)), columns=[f'X{i+1}' for i in range(p)])
data['y'] = (1
             + np.sin(data['X1']) * np.log1p(np.abs(data['X2']))
             + np.where(data['X5'] < 1, data['X3']**2 * data['X4'],0)
             + np.where(data['X3']>1, np.sqrt(np.abs(data['X5'])) * 5,0)
             + np.where(data['X6'] > 0, data['X6'], 0) ** 3) + np.random.normal(size=n)

# Splitting data into training and testing datasets
train_indices = np.random.choice(data.index, size=int(n * 0.8), replace=False)
DataTrain = data.loc[train_indices]
DataTest = data.drop(train_indices)

#Induce Sparsity
sparsity_level = 0.7
data.iloc[:, :-1] = data.iloc[:, :-1].mask(np.random.rand(*data.iloc[:, :-1].shape) < sparsity_level, 0)

# Gradient Boosting model setup
model = GradientBoostingRegressor(random_state=seed)

# Parameter space definition
param_space = {
    'max_depth': Integer(1, 10),
    'min_samples_split': Integer(2, 10),
    'learning_rate': Real(0.01, 0.2),
    'n_estimators': Integer(100, 2000)
}

# Bayes search with cross-validation
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1,
                              random_state=seed)
bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

# Best parameters and model evaluation
best_model = bayes_search.best_estimator_
predictions = best_model.predict(DataTest.drop('y', axis=1))
test_mse = mean_squared_error(DataTest['y'], predictions)

print("Best parameters found: ", bayes_search.best_params_)
print("Test MSE: ", test_mse)
