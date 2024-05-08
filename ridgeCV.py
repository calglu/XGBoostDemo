from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# Seed for reproducibility
seed = 42
np.random.seed(seed)

#Data generation
n = 1000
p = 50
data = pd.DataFrame(np.random.normal(size=(n, p)), columns=[f'X{i+1}' for i in range(p)])

#Induce Sparsity
sparsity_level = 0.7
data = data.mask(np.random.rand(*data.shape) < sparsity_level, 0)

data['y'] = (1
             + np.sin(data['X1']) * np.log1p(np.abs(data['X2']))
             + np.where(data['X5'] < 1, data['X3']**2 * data['X4'],0)
             + np.where(data['X3']>1, np.sqrt(np.abs(data['X5'])) * 5,0)
             + np.where(data['X6'] > 0, data['X6'], 0) ** 3) + np.random.normal(size=n)


# Splitting data into training and testing datasets
train_indices = np.random.choice(data.index, size=int(n * 0.8), replace=False)
DataTrain = data.loc[train_indices]
DataTest = data.drop(train_indices)


# AdaBoost model setup with Decision Tree as the base estimator
model = Ridge(random_state=seed)

# Parameter space for Ridge
ridge_param_space = {
    'alpha': Real(0.01, 100)
}


# Bayesian optimization setup for AdaBoost
ab_bayes_search = BayesSearchCV(estimator=model, search_spaces=ridge_param_space, n_iter=32, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1, random_state=seed)
ab_bayes_search.fit(DataTrain.drop('y', axis=1), DataTrain['y'])

# Evaluate the best AdaBoost model
ab_best_model = ab_bayes_search.best_estimator_
ab_predictions = ab_best_model.predict(DataTest.drop('y', axis=1))
ab_test_mse = mean_squared_error(DataTest['y'], ab_predictions)

print("AdaBoost - Best parameters found: ", ab_bayes_search.best_params_)
print("AdaBoost - Test MSE: ", ab_test_mse)
