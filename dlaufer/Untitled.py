import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import helper_functionspd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) 
get_ipython().run_line_magic('matplotlib', 'inline')

#load in files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#make copy dataframes
train_copy = train.copy()
test_copy = test.copy()

#Save the 'Id' column for submission and drop for target prediction
train_ID = train_copy['Id']
test_ID = test_copy['Id']
train_copy.drop('Id',axis=1,inplace=True)                      
test_copy.drop('Id',axis=1,inplace=True)

#remove SalePrice to do preprocessing

# Apply log transform since skewness is present
y_train = train_copy.SalePrice.values
train_copy.SalePrice = np.log1p(train_copy.SalePrice)
y_train_log = train_copy.SalePrice.values

#likely want log transform or johnson distribution transformation
train_copy.drop('SalePrice', axis=1, inplace=True)

#these need to be done separate because imputation here would be biased on test data if merging was done first

#coerce values to strings
str_vars = ['MSSubClass','YrSold','MoSold']
for var in str_vars:
    train_copy[var] = train_copy[var].apply(str)
    test_copy[var] = test_copy[var].apply(str)

# 'RL' is by far the most common value. So we can fill in missing values with 'RL'
train_copy['MSZoning'] = train_copy.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test_copy['MSZoning'] = test_copy.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
#there aren't any missing values here

train_copy['LotFrontage'] = train_copy.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_copy['LotFrontage'] = test_copy.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#merge train and test to do data preprocessing
data_features = pd.concat((train_copy, test_copy)).reset_index(drop=True)
print(data_features.shape)

#Overview of missingness of dataframe in total, by column and by rows
colpercent(data_features)

#Overview of classes within columns that have missingness
colpercount(data_features)

# Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

common_vars = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual', 'Functional']
for var in common_vars:
    data_features[var] = data_features[var].fillna(data_features[var].mode()[0])


# In[ ]:


#Overview of missingness of dataframe in total, by column and by rows
colpercent(data_features)


# In[ ]:


#Overview of classes within columns that have missingness
colpercount(data_features)


# In[9]:


# data description says NA means "No Pool", majority of houses have no Pool at all in general.
# features[] = features["PoolQC"].fillna("None")
# Replacing missing data with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"
           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
    data_features[col] = data_features[col].fillna('None')


# In[ ]:


#Overview of missingness of dataframe in total, by column and by rows
colpercent(data_features)


# In[ ]:


#Overview of classes within columns that have missingness
colpercount(data_features)


# In[10]:


# Replacing missing data with 0 (Since No garage = no cars in such garage.)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):
    data_features[col] = data_features[col].fillna(0)


# In[ ]:


#Overview of missingness of dataframe in total, by column and by rows
colpercent(data_features)


# In[ ]:


#Overview of classes within columns that have missingness
colpercount(data_features)


# In[11]:


# Calculating totals before droping less significant columns

#  Adding total sqfootage feature 
data_features['TotalSF']=data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']

#come back to this later
#  Adding total bathrooms feature
data_features['TotalBathrooms'] = (data_features['FullBath'] + (0.5 * data_features['HalfBath']) +
                               data_features['BsmtFullBath'] + (0.5 * data_features['BsmtHalfBath']))
#  Adding total porch sqfootage feature
data_features['TotalPorchSF'] = (data_features['OpenPorchSF'] + data_features['3SsnPorch'] +
                              data_features['EnclosedPorch'] + data_features['ScreenPorch'] +
                              data_features['WoodDeckSF'])

data_features = data_features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)

#data_features['YrBltAndRemod']=data_features['YearBuilt']+data_features['YearRemodAdd']


# In[ ]:


print(data_features['Utilities'].value_counts())
print(data_features['Street'].value_counts())
print(data_features['PoolQC'].value_counts())


# In[12]:


#dummify the features that can be dummified
data_features = pd.get_dummies(data=data_features, columns=data_features.dtypes[data_features.dtypes=="object"].index)


# In[ ]:


print('Features size:', data_features.shape)


# In[ ]:


print(len(data_features.columns))
print(data_features.dtypes.value_counts())
data_features


# In[13]:


#further stuff
#probably want to leave pool out outright
#data_features['HasPool'] = data_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['HasGarage'] = data_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['HasBsmt'] = data_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data_features['HasFireplace'] = data_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Not normaly distributed can not be normalised and has no central tendecy
# test leaving these in to make sure that the values don't contribute sufficient information content to merit leaving in
data_features = data_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)


# In[ ]:


#doesn't appear to be any negative values, empty strings or None types remaining
data_eval(data_features)


# In[147]:


from scipy.stats import shapiro

numeric_feats = data_features.dtypes[data_features.dtypes != "object"].index

test_normality = lambda x: shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(data_features[numeric_feats])
normal = normal.apply(test_normality)
print(not normal.any())

# # Check the skew of all numerical features
# skewed_feats = data_features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(10)


# In[14]:


#split back into train and test
train_copy = data_features.iloc[:len(y_train), :]
train_copy['SalePrice'] = y_train_log
test_copy = data_features.iloc[len(y_train):, :]
print(['Train data shape: ',train_copy.shape,'Prediction on (Sales price) shape: ', y_train.shape,'Test shape: ', test_copy.shape])


# In[15]:


# From EDA obvious outliers
train_copy = train_copy[train_copy.GrLivArea < 4500]
train_copy.reset_index(drop=True, inplace=True)
outliers = [30, 88, 462, 631, 1322]
train_copy = train_copy.drop(train_copy.index[outliers])


# In[ ]:


#need to work on cleaning these later
numerical_features   = train_copy.select_dtypes(exclude=['object']).columns.tolist()
categorical_features = train_copy.select_dtypes(include=['object']).columns.tolist()
#train_copy[categorical_features]
categorical_features


# In[ ]:


#need to figure out why this isn't working
#index_retrieve(train, 0.7, 'spearman')
print(train_copy.corr(method='spearman')['SalePrice'].sort_values(ascending=False)[1:30])
print('-'*35)
train_copy.corr()['SalePrice'].sort_values(ascending=False)[1:30]


# In[ ]:


#train_copy[numerical_features].hist()


# In[ ]:


#distribution of column data types (note object types coerced with presence of Nan, will need visual inspection later)

print(train_copy.dtypes.value_counts())
obj_list = list(train_copy.dtypes[train_copy.dtypes=="object"].index.values)
num_list = list(train_copy.dtypes[train_copy.dtypes!="object"].index.values)


# In[16]:


zero_columns_con = zeroper(train_copy, 99.5)
zero_columns_lib = zeroper(train_copy, 99)


# In[125]:


X = train_copy.drop('SalePrice', axis=1)
X_thin = X.drop(zero_columns_con, axis=1)
X_thinner = X.drop(column_drop, axis=1)
X_thinnest = X.drop(zero_columns_lib, axis=1)
X_test = test_copy
Y = train_copy.SalePrice


# In[ ]:


#X_thin.columns[:100]


# In[ ]:


len(X_thin.columns)


# In[18]:


from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge


# In[126]:


numerical_features   = X.select_dtypes(exclude=['object']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_features_thin   = X_thin.select_dtypes(exclude=['object']).columns.tolist()
categorical_features_thin = X_thin.select_dtypes(include=['object']).columns.tolist()

numerical_features_thinner   = X_thinner.select_dtypes(exclude=['object']).columns.tolist()
categorical_features_thinner = X_thinner.select_dtypes(include=['object']).columns.tolist()

numerical_features_thinnest   = X_thinnest.select_dtypes(exclude=['object']).columns.tolist()
categorical_features_thinnest = X_thinnest.select_dtypes(include=['object']).columns.tolist()


# In[127]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer,   numerical_features_thinner),
        ('cat', categorical_transformer, categorical_features_thinner)])


# In[21]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[128]:


# LinearRegression
pipe_Linear = Pipeline(
    steps   = [('preprocessor', preprocessor),
               ('Linear', LinearRegression()) ])    
# Ridge
pipe_Ridge = Pipeline(
    steps  = [('preprocessor', preprocessor),
              ('Ridge', Ridge(random_state=5)) ])  
# Huber
pipe_Huber = Pipeline(
    steps  = [('preprocessor', preprocessor),
              ('Huber', HuberRegressor()) ])  
# Lasso
pipe_Lasso = Pipeline(
    steps  = [ ('preprocessor', preprocessor),
               ('Lasso', Lasso(random_state=5)) ])
# ElasticNet
pipe_ElaNet = Pipeline(
    steps   = [ ('preprocessor', preprocessor),
                ('ElaNet', ElasticNet(random_state=5)) ])

# BayesianRidge
pipe_BayesRidge = Pipeline(
    steps   = [ ('preprocessor', preprocessor),
                ('BayesRidge', BayesianRidge(n_iter=500, compute_score=True)) ])

# clf = Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#   ('classification', RandomForestClassifier())
# ])


# In[54]:


###Model Lasso regression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = np.linspace(0.0005,1,250)).fit(X, Y)
rmse_cv(model_lasso).mean()


# In[80]:


lasso  = Lasso()
alphas = np.linspace(0.0001,0.001,50)
lasso.set_params(normalize=False)
coefs_lasso  = []
score_lasso = []

for alpha in alphas:
        lasso.set_params(alpha=alpha)
        lasso.fit(X, Y)
        score_lasso.append(rmse_cv(lasso.fit(X, Y)).mean())
        coefs_lasso.append(lasso.coef_)
        print('this is loop ', alpha)

coefs_lasso = pd.DataFrame(coefs_lasso, index = alphas, columns = X.columns)  


# In[82]:


for name in coefs_lasso.columns:
    plt.plot(coefs_lasso.index, coefs_lasso[name], label=name)
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')
plt.legend(loc=1)


# In[123]:


coefs_lasso.iloc[21:26].sum().value_counts(ascending=False)


# In[81]:


plt.plot(coefs_lasso.index, score_lasso)
plt.xlabel(r'hyperparameter $\lambda$')
plt.ylabel(r'slope values')
plt.legend(loc=1)


# In[124]:


coef = pd.Series(model_lasso.coef_, index = X.columns)
print(coef[coef==0].sort_values(ascending=False))
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
column_drop = coef[coef==0].sort_values(ascending=False).index


# In[47]:


#this indicates that MSZoning_C and SaleCondition_Abornmal need to go

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[129]:


list_pipelines = [pipe_Linear, pipe_Ridge, pipe_Huber, pipe_Lasso, pipe_ElaNet]

print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)
for pipe in list_pipelines :
    
    scores = cross_val_score(pipe, X_thinner, Y, scoring='neg_mean_squared_error', cv=5)
    scores = np.sqrt(-scores)
    print(pipe.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# In[130]:


list_scalers = [StandardScaler(), 
                RobustScaler(), 
                QuantileTransformer(output_distribution='normal')]
list_scalers = [StandardScaler()]


# In[131]:


parameters_Linear = { 'preprocessor__num__scaler': list_scalers,
                     'Linear__fit_intercept':  [True,False],
                     'Linear__normalize':  [True,False] }

gscv_Linear = GridSearchCV(pipe_Linear, parameters_Linear, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=0, cv=5)
gscv_Linear.fit(X_thinner, Y)

print(np.sqrt(-gscv_Linear.best_score_))
gscv_Linear.best_params_


# In[132]:


parameters_Ridge = { 'preprocessor__num__scaler': list_scalers,
                     'Ridge__alpha': [7,8,9],
                     'Ridge__fit_intercept':  [True,False],
                     'Ridge__normalize':  [True,False] }

gscv_Ridge = GridSearchCV(pipe_Ridge, parameters_Ridge, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=0, cv=5)
gscv_Ridge.fit(X_thinner, Y)

print(np.sqrt(-gscv_Ridge.best_score_))  
gscv_Ridge.best_params_


# In[133]:


parameters_Huber = { 'preprocessor__num__scaler': list_scalers,                   
                     'Huber__epsilon': [1.3, 1.35, 1.4],    
                     'Huber__max_iter': [150, 200, 250],                    
                     'Huber__alpha': [0.0005, 0.001, 0.002],
                     'Huber__fit_intercept':  [True], }

gscv_Huber = GridSearchCV(pipe_Huber, parameters_Huber, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_Huber.fit(X_thinner, Y)

print(np.sqrt(-gscv_Huber.best_score_))  
gscv_Huber.best_params_


# In[134]:


parameters_Lasso = { 'preprocessor__num__scaler': list_scalers,
                     'Lasso__alpha': [0.0005, 0.001],
                     'Lasso__fit_intercept':  [True],
                     'Lasso__normalize':  [True,False] }

gscv_Lasso = GridSearchCV(pipe_Lasso, parameters_Lasso, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_Lasso.fit(X_thinner, Y)

print(np.sqrt(-gscv_Lasso.best_score_))  
gscv_Lasso.best_params_


# In[135]:


parameters_ElaNet = { 'ElaNet__alpha': [0.0005, 0.001],
                      'ElaNet__l1_ratio':  [0.85, 0.9],
                      'ElaNet__normalize':  [True,False] }

gscv_ElaNet = GridSearchCV(pipe_ElaNet, parameters_ElaNet, n_jobs=-1, 
                          scoring='neg_mean_squared_error', verbose=1, cv=5)
gscv_ElaNet.fit(X_thinner, Y)

print(np.sqrt(-gscv_ElaNet.best_score_))  
gscv_ElaNet.best_params_


# In[136]:


list_pipelines_gscv = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]

print("model", "\t", "mean rmse", "\t", "std", "\t", "\t", "min rmse")
print("-+"*30)
for gscv in list_pipelines_gscv :
    
    scores = cross_val_score(gscv.best_estimator_, X_thinner, Y, 
                             scoring='neg_mean_squared_error', cv=5)
    scores = np.sqrt(-scores)
    print(gscv.estimator.steps[1][0], "\t", 
          '{:08.6f}'.format(np.mean(scores)), "\t",  
          '{:08.6f}'.format(np.std(scores)),  "\t", 
          '{:08.6f}'.format(np.min(scores)))


# In[137]:


linear_models = [gscv_Linear,gscv_Ridge,gscv_Huber,gscv_Lasso,gscv_ElaNet]


# In[138]:


pred_Linear = gscv_Linear.predict(X_test)
pred_Ridge  = gscv_Ridge.predict(X_test)
pred_Huber  = gscv_Huber.predict(X_test)
pred_Lasso  = gscv_Lasso.predict(X_test)
pred_ElaNet = gscv_ElaNet.predict(X_test)


# In[139]:


predictions_linear = {'Linear': pred_Linear, 'Ridge': pred_Ridge, 'Huber': pred_Huber,
                      'Lasso':  pred_Lasso, 'ElaNet': pred_ElaNet }


# In[141]:


predictions = {'Ridge': pred_Ridge, 'Lasso': pred_Lasso, 'ElaNet': pred_ElaNet}
df_predictions = pd.DataFrame(data=predictions_linear) 
df_predictions.corr()


# In[142]:


#this clearly doesn't work
pred_Blend_4 = (pred_Ridge + pred_Lasso + pred_ElaNet+ pred_Huber) / 4
sub_Blend_4 = pd.DataFrame()
sub_Blend_4['Id'] = test_ID
sub_Blend_4['SalePrice'] = np.expm1(pred_Blend_3)
sub_Blend_4.to_csv('Blend_Linear.csv',index=False)
sub_Blend_4.head()


# In[140]:


for model,values in predictions_linear.items():
    str_filename = model + ".csv"
    print("witing submission to : ", str_filename)
    subm = pd.DataFrame()
    subm['Id'] = test_ID
    subm['SalePrice'] = np.expm1(values)
    subm.to_csv(str_filename, index=False)


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

# In[ ]:


#levels of naively numeric type variables and their relative frequency

for i in range(len(num_list)):
    print('-'*55)
    print(train_copy[num_list[i]].value_counts())


# ## Helper Functions

# In[1]:


#helper function to print columns with missing and the percentage missingness:
def colpercent(df):
    print("Total NaN in Dataframe: " , df.isnull().sum().sum())
    print("Percent Missingness in Dataframe: ", 100*df.isnull().sum().sum()/(len(df.index)*len(df.columns)))
    print('-'*55)
    percentnulldf = df.isnull().sum()/(df.isnull().sum()+df.notna().sum())
    print("Percent Missingness by Columns:")
    print(100*percentnulldf[percentnulldf>0].sort_values(ascending=False))
    
#printout to help view levels within features with missingness
def colpercount(df):
    percentnulldf = df.isnull().sum()/(df.isnull().sum()+df.notna().sum())
    percent_ordered_df=percentnulldf[percentnulldf>0].sort_values(ascending=False)
    for i in range(len(percent_ordered_df)):
        print(percent_ordered_df.index[i])
        print('-'*15)
        print(train[percent_ordered_df.index[i]].value_counts())
        print('-'*55)

#helper function to print out percentage of zeroes by column

def zeroper(df, value):
    l=[]
    columns=[]
    for i in range(len(df.columns)):
        if 0 in df[df.columns[i]].value_counts():
            if 100*df[df.columns[i]].value_counts().loc[0]/len(df[df.columns[i]])>value:
                l.append((df.columns[i], 100*df[df.columns[i]].value_counts().loc[0]/len(df[df.columns[i]])))
            else:
                pass
        else:
            pass
    
    print(len(l))    
    print('-'*55)
    for j in range(len(l)):
        columns.append(l[j][0])
        print('Percent of zeroes: ', l[j])
        print('-'*55)
    print(columns)
    return columns
    
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
        if value<0:
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


# In[ ]:


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


# In[ ]:


#the script for removal

# col_remove = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
# train_copy.drop(col_remove, axis=1, inplace=True)


# In[ ]:


#the correlation matrix
train.corr()['SalePrice'].sort_values(ascending=False)


# In[ ]:


train_copy.corr()


# In[ ]:


#column position: train.corr().index[i]
#row position: len(train.corr().index)


# In[ ]:


sig_cor_index_list = index_retrieve(train_copy, 0.8, 'spearman')


# In[ ]:


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


clean_train = pd.read_csv('train_clean.csv')
train_copy = clean_train.copy()
df_features = clean_train.copy().drop('SalePrice', axis=1)


# In[ ]:


clean_train.head()


# In[ ]:


# HS_dummy = pd.get_dummies(df_features['HouseStyle'], prefix='HS', prefix_sep='__', drop_first = True)
# df_features = pd.concat([df_features.drop('HouseStyle', axis=1), HS_dummy], axis=1)
# EQ_dummy = pd.get_dummies(df_features['ExterQual'], prefix='EQ', prefix_sep='__', drop_first = True)
# df_features = pd.concat([df_features.drop('ExterQual', axis=1), EQ_dummy], axis=1)
# HQC_dummy = pd.get_dummies(df_features['HeatingQC'], prefix='HQC', prefix_sep='__',drop_first = True)
# df_features = pd.concat([df_features.drop('HeatingQC', axis=1), HQC_dummy], axis=1)
# KQ_dummy = pd.get_dummies(df_features['KitchenQual'], prefix='KQ', prefix_sep='__',drop_first = True)
# df_features = pd.concat([df_features.drop('KitchenQual', axis=1), KQ_dummy], axis=1)


# In[ ]:


# for i in range(len(train_copy.columns)):
#     if train_copy[train_copy.columns[i]].dtypes=="object":
#         var=''.join(c for c in str(train_copy.columns[i]) if c.isupper())[:2]
#         dummy= pd.get_dummies(train_copy[train_copy.columns[i]], prefix = var, prefix_sep='__',drop_first = True)
#         print(dummy)
#     else:
#         pass


# In[ ]:


# print(len(df_features.columns))
# df_features.head()


# In[ ]:


# ''.join(c for c in str(df_features.columns[i]) if c.isupper())[:2]
# str(df_features.columns[i]) + str('_dummy')


# In[ ]:


# columns=['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'KQ__Gd', 'KQ__TA',
#     'YearBuilt', 'LotArea', 'EQ__TA', 'OverallCond', 'BedroomAbvGr',
#     'GarageArea', 'MasVnrArea', 'Fireplaces', 'WoodDeckSF', 
#     'HQC__TA', 'HS__1.5Unf', 'HS__1Story', 'HS__2.5Fin', 'HS__2.5Unf',
#     'HS__2Story', 'HS__SFoyer', 'HS__SLvl', 'TotRmsAbvGrd']

# df_fin_features = df_features[columns]


# In[ ]:


df_fin_features = df_fin_features.fillna(0)


# In[ ]:


print(len(df_fin_features.columns))
df_fin_features.head()


# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


train_copy.corr().SalePrice.sort_values()


# In[ ]:


for i in range(len(df_fin_features.columns)):
    if 0 in df_fin_features[df_fin_features.columns[i]].value_counts():
        print('-'*55)
        print('Percentage zeros in column ' + str(i) + ', ' + str(df_fin_features.columns[i]) ': ' + df_fin_features[df_fin_features.columns[i]].value_counts().loc[0]/len(df_fin_features[df_fin_features.columns[i]]))
    else:
        pass


# In[ ]:


print('Percentage zeros in column ' + str(1) + ', ' + str(df_fin_features.columns[1]) ': ' + df_fin_features[df_fin_features.columns[1]].value_counts().loc[0]/len(df_fin_features[df_fin_features.columns[1]]))


# In[ ]:


for i in range(len(df_fin_features.columns)):
    if 0 in df_fin_features[df_fin_features.columns[i]].value_counts():
        print('-'*55)
        print(df_fin_features.columns[i])
        print('Percent of zeroes: ', 100*df_fin_features[df_fin_features.columns[i]].value_counts().loc[0]/len(df_fin_features[df_fin_features.columns[i]]))
    else:
        pass


# In[ ]:


print("Total NaN in Dataframe: " , df_fin_features.isnull().sum().sum())
print("Percent Missingness in Dataframe: ", 100*df_fin_features.isnull().sum().sum()/(len(df_fin_features.index)*len(df_fin_features.columns)))
print('-'*55)
percentnulldf = df_fin_features.isnull().sum()/(df_fin_features.isnull().sum()+df_fin_features.notna().sum())
print("Percent Missingness by Columns:")
100*percentnulldf[percentnulldf>0].sort_values(ascending=False)


# In[ ]:


elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, normalize=True)
N_alpha = 100
N_rho   = 10
alphaRange = np.logspace(-10, 2, N_alpha)
rhoRange   = np.linspace(0.1,1, N_rho) # we avoid very small rho by starting at 0.1
scores     = np.zeros((N_rho, N_alpha))
prices = pd.Series(clean_train.SalePrice)
for alphaIdx, alpha in enumerate(alphaRange):
    for rhoIdx, rho in enumerate(rhoRange):
            elasticnet.set_params(alpha=alpha, l1_ratio=rho)
            elasticnet.fit(df_fin_features, prices)
            scores[rhoIdx, alphaIdx] = elasticnet.score(df_fin_features, prices)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()

params = {'alpha':(0.1, 0.5, 1, 2), 'fit_intercept':[True,False]}
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=params)
grid_search_lasso.fit(Xtrain, ytrain)

grid_search_lasso.best_params_

grid_search_lasso.best_score_


# In[ ]:


print((train_copy['GrLivArea']-train_copy['LowQualFinSF'] + train_copy['TotalBsmtSF']).corr(train_copy['SalePrice'], method='spearman'))
(train_copy['GrLivArea']).corr(train_copy['SalePrice'], method='spearman')
train_copy['TotSF'] = train_copy['GrLivArea']-train_copy['LowQualFinSF'] + train_copy['TotalBsmtSF']

#train_copy['SalePrice']
#train_copy.columns


# In[ ]:


train_copy.columns


from scipy.stats import skew, kurtosis

for i in range(len(df_fin_features.columns)):
    print('Skewness and Kurtosis for ' + str(df_fin_features.columns[i]) + ':')
    print(df_fin_features[df_fin_features.columns[i]].skew(),"   ", df_fin_features[df_fin_features.columns[i]].kurtosis())
    if np.log1p(df_fin_features[df_fin_features.columns[i]])[np.log1p(df_fin_features[df_fin_features.columns[i]])!=0].skew() == 0:
        print('Dummified Feature, Please Ignore')
    else:
        print(np.log1p(df_fin_features[df_fin_features.columns[i]])[np.log1p(df_fin_features[df_fin_features.columns[i]])!=0].skew(), '  ', np.log1p(df_fin_features[df_fin_features.columns[i]])[np.log1p(df_fin_features[df_fin_features.columns[i]])!=0].kurtosis())


df_fin_features.dropna().hist(bins=50, figsize=(20,15))




