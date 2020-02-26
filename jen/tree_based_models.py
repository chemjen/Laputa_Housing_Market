import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble

data = pd.read_csv('../train_clean.csv')

categoricals = data.columns[data.dtypes == object]
lencoder = LabelEncoder()
for i, column in enumerate(categoricals):
	if i == 0:
		categorical_array = lencoder.fit_transform(data[column]).reshape((-1,1))
	else:
		categorical_array = np.concatenate((categorical_array,
			lencoder.fit_transform(data[column]).reshape((-1,1))), axis=1)

categorical_df = pd.DataFrame(categorical_array, columns=categoricals)

features = pd.concat([data.drop(categoricals,axis=1), categorical_df], axis=1)
features.drop('SalePrice', axis=1, inplace=True)
target = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(features, target, 
	test_size=0.2, random_state=0)

randomForest = ensemble.RandomForestRegressor()
gbm = ensemble.GradientBoostingRegressor()

# {'max_depth': 13, 'max_features': 0.02, 'min_samples_leaf': 5, 
# 'n_estimators': 70}

#grid_para_forest = {
#    'max_depth': range(25,43, 1),
#    'n_estimators': range(100,150,10),
#    'min_samples_leaf': np.array([1, 2, 3]),
#    'max_features': np.arange(0.2, 0.38, 0.02)
#}

#grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, 
#	scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
#grid_search_forest.fit(features, target)
#print(grid_search_forest.best_params_)

#randomForest.set_params(max_depth=20)
#{'max_depth': 36, 'max_features': 0.026, 'min_samples_leaf': 1, 'n_estimators': 120}
#{'max_depth': 38, 'max_features': 0.028, 'min_samples_leaf': 1, 'n_estimators': 110}
#{'max_depth': 34, 'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 120}

import XGBoost



