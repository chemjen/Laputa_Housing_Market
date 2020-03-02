import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump
import sys

model = sys.argv[1]
if model == 'gradboost':
	from sklearn.ensemble import GradientBoostingRegressor
	reg = GradientBoostingRegressor()
	grid_params = {'n_estimators': range(50,120,10)}
elif model == 'randomforest':
	from sklearn.ensemble import RandomForestRegressor
	reg = RandomForestRegressor()
	grid_params = { 'max_depth': range(30,45, 1),
		'n_estimators': range(60,150,10)}
elif model == 'xgboost':
	from xgboost import XGBRegressor
	reg =  XGBRegressor()
	grid_params = {'max_depth': range(1,4,1), 'n_estimators': 
		range(300,500,20)}
else:
	raise NameError(model+' is not a tree regression model')

version = sys.argv[2]
if version == 'v1':
	data = pd.read_csv('../train_clean.csv')
else:
	try:
		data = pd.read_csv('../train_clean_'+version+'.csv')
	except:
		raise NameError(version+' is not a version')
		
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
print('rawr!')
grid_search_reg  = GridSearchCV(reg, grid_params, scoring='r2', cv=5, 
	n_jobs=-1)
grid_search_reg.fit(features, target)
print(grid_search_reg.best_params_)

print(grid_search_reg.best_score_)
best_est = grid_search_reg.best_estimator_

print(best_est.score(features, target))
dump(best_est, './estimators/'+model+'_'+version+'.joblib')

