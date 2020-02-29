import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import OneHotEncoder
#from sklearn import ensemble
import xgboost as xgb

data = pd.read_csv('../train_clean.csv')

categoricals = data.columns[data.dtypes == object]

for i, column in enumerate(categoricals):
	if i ==0:
		cat_df = pd.get_dummies(data[column], prefix=column)
	else:
		cat_df = pd.concat([cat_df, pd.get_dummies(data[column], 
			prefix=column)], axis=1)

data_processed = pd.concat([data.drop(categoricals, axis=1), cat_df], axis=1)

#print(data_processed)
target = data_processed['LogSalePrice']
features = data_processed.drop(['LogSalePrice', 'SalePrice'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3) 
#	test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix( X_test) #, label=y_test)


#param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
param = {'objective': 'reg:squarederror'}
#param['nthread'] = 4
#param['eval_metric'] = 'auc'

#evallist = [(dtest, 'eval'), (dtrain, 'train')]
#num_round = 10
#bst = xgb.train(param, dtrain) #, num_round, evallist)
#print(bst)
# https://xgboost.readthedocs.io/en/latest/parameter.html
#ypred = bst.predict(dtest)

#print(ypred)

xgb.XGBRegressor(X_train, y_train)





