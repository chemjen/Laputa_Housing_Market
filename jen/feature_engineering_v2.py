import pandas as pd
import numpy as np

datadir = "../house-prices-advanced-regression-techniques/"
train = pd.read_csv(datadir+"train.csv")
test = pd.read_csv(datadir+"test.csv")

train = train.loc[train['GrLivArea'] < 4500] # outlier removal

for df in [train, test]:

	df.set_index("Id", inplace=True)
	df['KitchenQual'] = df['KitchenQual'].fillna('TA')
	df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
	df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
	df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
	df['GarageArea'] = df['GarageArea'].fillna(0)
	df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
	df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + \
		df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
	df['FullBath']  = df['FullBath'] + df['BsmtFullBath']
	df['HalfBath'] = df['HalfBath'] + df['BsmtHalfBath']
	df['TotSF'] = df['GrLivArea']-df['LowQualFinSF'] + df['TotalBsmtSF']
	df.drop(['FireplaceQu', 'Utilities', 'MasVnrType', 'LotFrontage', 
		'Functional', 'Fence', 'Electrical', 'Alley', 'BsmtFinType2', 'GarageCond',
		'MSZoning', 'MiscFeature', 'BsmtFinSF2', 'SaleType', 'BsmtCond', 
		'GarageQual', 'PoolQC', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtQual',
		'BsmtExposure','BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF','Exterior1st', 
		'Exterior2nd', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars'], 
		axis=1, inplace=True) 

## performed a log transform of SalePrice
## https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
train['LogSalePrice'] = np.log1p(train['SalePrice'])
train.to_csv('../train_clean_v2.csv', index=False)
test.to_csv('../test_clean_v2.csv', index=False)

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
