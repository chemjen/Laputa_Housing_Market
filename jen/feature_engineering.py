import pandas as pd
import numpy as np

datadir = "../house-prices-advanced-regression-techniques/"
train = pd.read_csv(datadir+"train.csv")
test = pd.read_csv(datadir+"test.csv")

for df in [train, test]:

	df.set_index("Id", inplace=True)
	df['KitchenQual'] = df['KitchenQual'].fillna('TA')
	df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
	df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
	df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
	df['GarageArea'] = df['GarageArea'].fillna(0)
	df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)

	df['FullBath']  = df['FullBath'] + df['BsmtFullBath']
	df['HalfBath'] = df['HalfBath'] + df['BsmtHalfBath']

	df.drop(['FireplaceQu', 'Street', 'Utilities', 'LandContour', 
		'MasVnrType', 'Condition2', 'PoolArea', 'LotFrontage', 'CentralAir',
		'Functional', 'LandSlope', 'LotConfig', 'Fence', 'BldgType', 'Street',
		'Electrical', 'Alley', 'RoofStyle', 'KitchenAbvGr', 'BsmtFinType2',
		'Heating', 'PavedDrive', 'LandContour', 'Condition1', 'GarageCond', 
		'ExterCond', 'MSZoning', 'MiscFeature', 'SaleCondition', 'BsmtFinSF2',
		'SaleType', 'BsmtCond', 'MiscVal', 'GarageQual','EnclosedPorch',
		'3SsnPorch', 'RoofMatl', 'ScreenPorch', 'LowQualFinSF', 'Condition2',
		'PoolQC', 'PoolArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtQual',
		'BsmtExposure','BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', 
		'2ndFlrSF', 'MSSubClass', 'LotShape', 'Neighborhood', 'YearRemodAdd',
		'Exterior1st', 'Exterior2nd', 'GarageType', 'GarageYrBlt', 'Foundation',
		'GarageFinish', 'GarageCars'], axis=1, inplace=True) 

train = train.loc[train['GrLivArea'] < 4500] # outlier removal

## performed a log transform of SalePrice
## https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
train['LogSalePrice'] = np.log1p(train['SalePrice'])
train.to_csv('../train_clean.csv', index=False)
test.to_csv('../test_clean.csv', index=False)


## 

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
