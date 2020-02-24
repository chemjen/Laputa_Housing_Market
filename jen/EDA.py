import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
train = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

#for column in train.columns:
#    percentages = train[column].value_counts()/1460 #.values.flatten()/1460
#    if percentages.iloc[0] > 0.7:
#        print(column)
#        print(train.groupby(column)['SalePrice'].mean())
train.fillna(0, inplace=True)
stds = {} 
for column in train.columns: 
    percentages = train[column].value_counts()/1460 #.values.flatten()/1460 
    if percentages.iloc[0] > 0.7: 
        stds[column] = np.std(train.groupby(column)['SalePrice'].mean().values.flatten())                                                                                                                         

print({k: v for k, v in sorted(stds.items(), key=lambda item: item[1])})


pvals = {}
for column in train.columns: 
    percentages = train[column].value_counts()/1460 #.values.flatten()/1460 
    if percentages.iloc[0] > 0.7: 
        pval = ttest_ind(train.loc[train[column]==percentages.index[0]]['SalePrice'].values.flatten(),
           train.loc[~(train[column]==percentages.index[0])]['SalePrice'].values.flatten()).pvalue
        if pval < 0.05:
            pvals[column] = pval

print({k: v for k, v in sorted(pvals.items(), key=lambda item: item[1])})

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



