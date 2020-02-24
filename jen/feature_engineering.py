import pandas as pd
import numpy as np

train = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv")
train.columns
train.set_index("Id", inplace=True)

train = train.loc[train['GrLivArea'] < 4500] # outlier removal

train['FullBath'] = train['FullBath'] + train['BsmtFullBath']
train['HalfBath'] = train['HalfBath'] + train['BsmtHalfBath']

train.drop(['FireplaceQu', 'Street', 'Utilities', 'LandContour', 'MasVnrType',
	'Condition2', 'PoolArea', 'LotFrontage', 'CentralAir', 'Functional',
	'LandSlope', 'LotConfig', 'Fence', 'BldgType', 'Street', 'Electrical',
	'Alley', 'RoofStyle', 'KitchenAbvGr', 'BsmtFinType2', 'Heating',
	'PavedDrive', 'LandContour', 'Condition1', 'GarageCond', 'ExterCond',
	'MSZoning', 'MiscFeature', 'SaleCondition', 'BsmtFinSF2', 'SaleType',
	'BsmtCond', 'MiscVal', 'GarageQual','EnclosedPorch','3SsnPorch',
	'RoofMatl', 'ScreenPorch', 'LowQualFinSF', 'Condition2', 'PoolQC', 
	'PoolArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtQual', 'BsmtExposure',
	'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF'], 
	axis=1, inplace=True)

train.drop(['MSSubClass', 'LotShape', 'Neighborhood', 'YearRemodAdd',
    'Exterior1st', 'Exterior2nd', 'GarageType', 'GarageYrBlt', 'Foundation',
	'GarageFinish', 'GarageCars'], axis=1, inplace=True)

train.to_csv('../train_clean.csv', index=False)
## 
print(train.columns)
#for column in train.columns:
#    percentages = train[column].value_counts()/1460 #.values.flatten()/1460
#    if percentages.iloc[0] > 0.7:
#        print(column)
#        print(train.groupby(column)['SalePrice'].mean())

#stds = {} 
#   for column in train.columns: 
#       percentages = train[column].value_counts()/1460 #.values.flatten()/1460 
#       if percentages.iloc[0] > 0.7: 
#           print(column) 
#           print(percentages.head(1)) 
#           stds[column] = np.std(train.groupby(column)['SalePrice'].mean().values.flatten())                                                                                                                         

#{k: v for k, v in sorted(stds.items(), key=lambda item: item[1])}      

#from scipy.stats import ttest_ind
#for column in train.columns:
#    percentages = train[column].value_counts()/1460 #.values.flatten()/1460 
#    if percentages.iloc[0] > 0.7:
#        pval = ttest_ind(train.loc[train[column]==percentages.index[0]]['SalePrice'].values.flatten(),
#        train.loc[~(train[column]==percentages.index[0])]['SalePrice'].values.flatten()).pvalue
#        if pval < 0.05:
#            print(column, pval)
#for column in train.columns:
#    percentages = train[column].value_counts()/1460 #.values.flatten()/1460 
#    if percentages.iloc[0] > 0.8:
#        pval = ttest_ind(train.loc[train[column]==percentages.index[0]]['SalePrice'].values.flatten(),
#        train.loc[~(train[column]==percentages.index[0])]['SalePrice'].values.flatten()).pvalue
#        if pval < 0.05:
#            print(column, pval)
