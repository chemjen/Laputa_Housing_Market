#!/usr/bin/env python
# coding: utf-8

# # Initial EDA on Zillow Data

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 80)


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train_copy = train.copy()
test_copy = test.copy()


# In[4]:


train


# In[5]:


#distribution of column data types (note object types coerced with presence of Nan, will need visual inspection later)

print(train.dtypes.value_counts())
obj_list = list(train.dtypes[train.dtypes=="object"].index.values)
num_list = list(train.dtypes[train.dtypes!="object"].index.values)


# In[6]:


#the train and test columns differ in the number of columns they contain, as expected with columns

print(train.shape)
print(test.shape)


# In[7]:


#levels of naively categorical bits and their relative frequency

#train[obj_list]
#print(obj_list)

for i in range(len(obj_list)):
    print('-'*55)
    print(train[obj_list[i]].value_counts())
    print('Number of missing values in column: ', train[obj_list[i]].isnull().sum())


# Features that count the number of rooms. This seems to be more categorical as you can't have 3.245 bedrooms for example.
# 
# The same holds for the number of fireplaces and garage cars.
# 
# Features like OverallCond and OverallQual can be considered as groups and are categorical variables with a natural kind of order (ordinal features).
# 
# Every feature with SF or area can be considered as numerical.
# 
# I think that we should treat temporal features like YrSold as categorical.
# 
# But what about MisVal and 3SsnPorch? Reading in the description, we can see that the latter is related to area. The further stands for the value in $ of a miscellaneous feature like a tennis court. Consequently both are numerical.
# 
# Comparing with the description this looks fine! It seems that there are no numerical features in categorical candidates. Some categorical features might still be completely useless as only some levels occur most of the time (no diversity). To get more insights, we can compute the frequency of the most common level in the train & test data:
# 
# Many categorical candidates have one major level that occupies > 80 % 
# of the data. That's bad. What should we learn from such a feature? That they lack level diversity and therefore don't contribute much useful information to the model for the purposes of prediction.

# In[8]:


#levels of naively numeric type variables and their relative frequency

for i in range(len(num_list)):
    print('-'*55)
    print(train[num_list[i]].value_counts())


# In[9]:


#summary statistics after nan removal of numeric features
train[num_list].describe()


# ## Overview of missingness of dataframe in total, by column and by rows

# In[10]:


#Overview of missingness of dataframe in total, by column and by rows

print("Total NaN in Dataframe: " , train.isnull().sum().sum())
print("Percent Missingness in Dataframe: ", 100*train.isnull().sum().sum()/(len(train.index)*len(train.columns)))
print('-'*55)
percentnulldf = train.isnull().sum()/(train.isnull().sum()+train.notna().sum())
print("Percent Missingness by Columns:")
100*percentnulldf[percentnulldf>0].sort_values(ascending=False)


# Degree of missingness is small but structurally potentially significant.
# 
# We can see that PoolQC, MiscFeature, Alley etc. have more than 80 % missing values. 
# In the description, we can see that this often tells us "no pool", "no miscfeature" etc.
# In my opinion it's difficult to say if such a feature is structurally important or not. 
# Let's not drop them for the moment, either plugin "None", impute, or remove later in the analysis
# 
# running list of features to potentially remove based on missingness: PoolQC, MiscFeature, Alley, Fence
# features to consider removing based on missingness: FireplaceQu

# In[11]:


#printout to help view levels within features with missingness

percent_ordered_df=percentnulldf[percentnulldf>0].sort_values(ascending=False)
for i in range(len(percent_ordered_df)):
    print(percent_ordered_df.index[i])
    print('-'*15)
    print(train[percent_ordered_df.index[i]].value_counts())
    print('-'*55)


# In[34]:


#helper functions to characterize missingness by row and column

def data_eval(df):
    for i in range(len(df.columns)):
        print('-'*50)
        print('Column Name: ', df.columns[i])
        if (df[df.columns[i]].dtypes == 'float64' or df[df.columns[i]].dtypes == 'int64') and df[df.columns[i]][df[df.columns[i]]<0].count()>0:
            print('Number of negatives: ', df[df.columns[i]][df[df.columns[i]]<0].count())
        if df[df.columns[i]][df[df.columns[i]]=='None'].count() > 0:
            print('Number of None strings: ', df[df.columns[i]][df[df.columns[i]]=='None'].count())
        if df[df.columns[i]][df[df.columns[i]]==''].count() > 0:
            print('Number of empty strings: ', df[df.columns[i]][df[df.columns[i]]==''].count())
        else:
            print('Column ' + str(i) + ' has no negatives, empty strings or Nones')
#     for i in range(len(df.index)) :
#         print("Nan in row ", i , " : " ,  df.iloc[i].isnull().sum())

              
# def row_eval(df, value):
#     for i in range(len(df.index)) :
#         print('-'*50)
#         print('Row Index: ', i)
#         print('Number of nan in row: ', df.iloc[i].isnull().sum())
#         print(df.iloc[i].isnull().sum(), df.iloc[i].isnull().sum() > value)

def row_na_list(df, value):
    l=[]
    for i in range(len(df.index)) :
        if df.iloc[i].isnull().sum() > value:
            l.append([i, df.iloc[i].isnull().sum()])
    return l

#helper function to retrieve row and column index labels for correlation matrix values
#for greater than value when value>0 and less than value when value<0
#and prints out the values that correspond to those indices

def index_retrieve(df, value, measure):
    ''' Get index positions of value in dataframe.'''

    poslist = list()
    # Get bool dataframe with True at positions where the given value exists and filter out on-diagonal elements
    if measure == 'spearman':
        if value>0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)>value].isna().isin([value])
        elif value<0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)<value].isna().isin([value])
        else:
            pass
    elif measure == 'pearson':
        if value>0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)>value].isna().isin([value])
        elif value<0:
            result = df.corr(method = measure)[df.corr(method = measure)!=1][df.corr(method = measure)<value].isna().isin([value])
        else:
            pass
    # Get list of columns that contains the value
    series = result.any()
    columnNames = list(series[series == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            poslist.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    
    if value > 0:
        print('Number of correlations with value greater than ' + str(value) + ': ' + str(len(poslist)))
    if value < 0:
        print('Number of correlations with value less than ' + str(value) + ': ' + str(len(poslist)))
    else:
        pass
    for i in range(len(poslist)):
        print('-'*40)
        print('index labels: ', poslist[i][0], poslist[i][1])
        print('value at index: ', df.corr().loc[poslist[i]])
    
    return poslist


# In[13]:


#there are not many rows with more than 10 nan's for features. The percent missingness is relatively low.
#more testing will be required to see if model performance improves once
#1. features are reduced and cleaned, and 
#2. removal of high missingness observations
#3. if imputation will be more effective rather than removal

value=10
row_na = row_na_list(train,value)
print('Number of Rows with missingness greater than ' + str(value) + ': ' + str(len(row_na)))
print('Minimum percentage missingness: ', 100*(value+1)/len(train.columns))
for i in range(len(row_na)):
    print('-'*50)
    print('Row ' + str(row_na[i][0]) + ' with ' +str(100*row_na[i][1]/len(train.columns)) + ' percent missingness')
    print('Number of missing entries in row: ', row_na[i][1])


# In[14]:


#there are no negative values in numeric columns (object columns still need to be inspected)
#also no empty strings
#there are None strings in 
data_eval(train)


# In[15]:


#the script for removal

# col_remove = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
# train_copy.drop(col_remove, axis=1, inplace=True)


# In[16]:


#train.Electrical[train['Electrical']=='None'].count()
# for i in range(len(train.columns)):
#     print('-'*55)
#     print('Column name: ', train.columns[i])
#     print('Number of None strings: ', train[train.columns[i]][train[train.columns[i]]=='None'].count())
#     print('Number of empty strings: ', train[train.columns[i]][train[train.columns[i]]==''].count())


# In[17]:


#plausible to binarize Electrical into SBrkr and other and imputate by mean on feature's NA's
print('percent of SBrkr electrical hookups of total: ', len(train.Electrical[train['Electrical']=='SBrkr'])/len(train['Electrical']))
print(train.Electrical.notna().value_counts())
train[train.Electrical.isna()]


# In[27]:


#the correlation matrix
train.corr()


# In[28]:


#column position: train.corr().index[i]
#row position: len(train.corr().index)


# In[36]:


sig_cor_index_list = index_retrieve(train, 0.7, 'spearman')


# In[40]:


print(train.corr(method='spearman').SalePrice.sort_values(ascending=False))
train.corr().SalePrice.sort_values(ascending=False)


# ## Important Notes

# Useful links: 
# https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding
# 

# 1. Scikit-learn is sensitive to the ordering of columns, so if the training dataset and test datasets get misaligned, your results will be nonsense. This could happen if a categorical had a different number of values in the training data vs the test data. Ensure the test data is encoded in the same manner as the training data with the align command
# 
# 2. It's most common to one-hot encode these "object" columns, since they can't be plugged directly into most models. Pandas offers a convenient function called get_dummies to get one-hot encodings. Alternatively, you could have dropped the categoricals. To see how the approaches compare, we can calculate the mean absolute error of models built with two alternative sets of predictors (One-hot encoding usually helps, but it varies on a case-by-case basis):
# 
#     One-hot encoded categoricals as well as numeric predictors
#     Numerical predictors, where we drop categoricals.
#     
# 3. Potentially using Lasso regression to perform feature selection. Must normalize or standardize features prior to performing regression

# In[ ]:





# In[18]:


#
# print(0)
# print('percent of SBrkr electrical hookups of total: ', len(train.MasVnrType[train['Electrical']=='SBrkr'])/len(train['Electrical']))
# train.MasVnrType.notna().value_counts()

