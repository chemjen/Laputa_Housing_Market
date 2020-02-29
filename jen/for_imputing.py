import pandas as pd
import numpy as np

train = pd.read_csv("../house-prices-advanced-regression-techniques/test.csv")
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

train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train.isnull().sum(axis=0)
test = train
test.isnull().sum(axis=0)
testorig = pd.read_csv('../house-prices-advanced-regression-techniques/test.csv')
testorig.loc[testorig['TotalBasementSF'].isnull()]
testorig.loc[testorig['TotalBsmtSF'].isnull()]
testorig.loc[testorig['TotalBsmtSF'].isnull()].T
testorig.loc[testorig['TotalBsmtSF'].isnull()].T[:10]
testorig.loc[testorig['TotalBsmtSF'].isnull()].T[:30]
testorig.loc[testorig['TotalBsmtSF'].isnull()].T[30:60]
testorig.loc[testorig['TotalBsmtSF'].isnull()].T[60:90]
testorig.loc[testorig['KitchenQual'].isnull()].T[60:90]
testorig.loc[testorig['KitchenQual'].isnull()].T[:40]
testorig.loc[testorig['KitchenQual'].isnull()].T[40:80]
testorig.loc[testorig['GarageArea'].isnull()].T[40:80]
test.isnull().sum(axis=0)
testorig.loc[testorig['GarageArea'].isnull()].T[40:80]
test.isnull().sum(axis=0)
# TotalBsmtSF = 0, KitchenQual=TA, GarageArea=1
train = pd.read_csv('./Laputa_Housing_Market/house-prices-advanced-regression-techniques/train.csv')
train = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')
train.columns
test.isnull().sum(axis=0)
testorig.isnull().sum(axis=0)
testorig.isnull().sum(axis=0)[:40]
testorig.isnull().sum(axis=0)[40:]
testorig.loc[testorig['BsmtFullBath'].isnull()].T[40:80]
testorig.loc[testorig['BsmtFullBath'].isnull()].T[:40]
testorig.loc[testorig['BsmtHalfBath'].isnull()].T[:40]
test.isnull().sum(axis=0)
history
