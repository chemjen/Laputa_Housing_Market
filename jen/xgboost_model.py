import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../train_clean_v2.csv')

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
features.drop(['LogSalePrice', 'SalePrice'], axis=1, inplace=True)
target = data['LogSalePrice']

#X_train, X_test, y_train, y_test = train_test_split(features, target, 
#	test_size=0.2, random_state=0)
#gbm = ensemble.GradientBoostingRegressor()

from xgboost import XGBRegressor
xgb = XGBRegressor()

#xgb.fit(X_train, y_train)
#print(xgb.feature_importances_)

grid_para_xgb = {'max_depth': range(1,4,1), 'n_estimators': range(300,500,20)}
xgb.set_params(verbosity=0)

grid_search_xgb = ms.GridSearchCV(xgb, grid_para_xgb, #scoring='neg_mean_squared_error', 
	scoring='r2',
	cv=5, n_jobs=-1)
grid_search_xgb.fit(features, target)
print(grid_search_xgb.best_params_)

print(grid_search_xgb.best_score_)
# newdf = pd.DataFrame(grid_search_xgb.best_estimator_.feature_importances_, 
# index=X_train.columns, columns='importance')

best_est = grid_search_xgb.best_estimator_

print(best_est.score(features, target))

newdf = pd.DataFrame(grid_search_xgb.best_estimator_.feature_importances_, 
	index=features.columns, columns=['importance'])
newdf.sort_values(by='importance', ascending=False).plot.bar()
plt.tight_layout()
plt.show()

test = pd.read_csv('../test_clean_v2.csv')
for i, column in enumerate(categoricals):
	if i == 0:
		categorical_array = lencoder.fit_transform(test[column]).reshape((-1,1))
	else:
		categorical_array = np.concatenate((categorical_array,
			lencoder.fit_transform(test[column]).reshape((-1,1))), axis=1)

categorical_df = pd.DataFrame(categorical_array, columns=categoricals)

test = pd.concat([test.drop(categoricals,axis=1), categorical_df], axis=1)

predictions = best_est.predict(test)
predictions = np.expm1(predictions)
print(predictions)

predictions = pd.DataFrame(data=predictions, columns=['SalePrice'])
predictions['Id'] = np.arange(1461, 1461+len(predictions))

predictions.to_csv('xgboost_predictions_v2.csv', index=False)







