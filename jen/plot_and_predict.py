import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from joblib import load
import sys

model = sys.argv[1] # gradboost, xgboost, randomforest
version = sys.argv[2]  # 'v1', 'v2'

if version == 'v1':
	test = pd.read_csv('../test_clean.csv')
else:
	test = pd.read_csv('../test_clean_'+version+'.csv')

best_est = load('./estimators/gradboost_'+version+'.joblib')
newdf = pd.DataFrame(best_est.feature_importances_, index=test.columns, 
	columns=['importance'])
newdf.sort_values(by='importance', ascending=False).plot.bar()
plt.title('Gradient Boost')
plt.tight_layout()
plt.show()

categoricals = test.columns[test.dtypes == object]
lencoder = LabelEncoder()
for i, column in enumerate(categoricals):
	if i == 0:
		categorical_array = lencoder.fit_transform(test[column]).\
			reshape((-1,1))
	else:
		categorical_array = np.concatenate((categorical_array,
			lencoder.fit_transform(test[column]).reshape((-1,1))), axis=1)

categorical_df = pd.DataFrame(categorical_array, columns=categoricals)

test = pd.concat([test.drop(categoricals,axis=1), categorical_df], axis=1)

predictions = best_est.predict(test)
predictions = np.expm1(predictions)

predictions = pd.DataFrame(data=predictions, columns=['SalePrice'])
predictions['Id'] = np.arange(1461, 1461+len(predictions))

predictions.to_csv('./predictions/'+model+'_predictions_'+version+'.csv', 
	index=False)







