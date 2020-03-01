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
features.drop(['LogSalePrice', 'SalePrice'], axis=1, inplace=True)
target = data['LogSalePrice']

gbm = ensemble.GradientBoostingRegressor()

grid_para_gbm = {'n_estimators': range(50,120,10)}

grid_search_gbm = ms.GridSearchCV(gbm, grid_para_gbm, #scoring='neg_mean_squared_error', 
	scoring='r2', cv=5, n_jobs=-1)
grid_search_gbm.fit(features, target)
print(grid_search_gbm.best_params_)
#quit()
print(grid_search_gbm.best_score_)
best_est = grid_search_gbm.best_estimator_

print(best_est.score(features, target))

newdf = pd.DataFrame(grid_search_gbm.best_estimator_.feature_importances_, 
	index=features.columns, columns=['importance'])
newdf.sort_values(by='importance', ascending=False).plot.bar()
plt.tight_layout()
plt.show()

test = pd.read_csv('../test_clean.csv')
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

predictions.to_csv('gradboost_predictions.csv', index=False)







