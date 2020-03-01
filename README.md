# Laputa_Housing_Market
This is Team Laputa kaggle competition repo from the housing prices
competition: 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Laputa comes from the name of the castle from Castle in the Sky
Team Members:
Hanbo Shao: shimmer-croissant0707
Liliana Nishihira: lnishihira
Dan Laufer: DanLauferPhysics
Jen Ruddock: chemjen

--------------------------------------
./jen/

This directory contains Jen's files on tree based models, feature engineering,
and also some EDA.

EDA: EDA_initial.py and for_imputation.py

The feature_engineering.py and feature_engineering_v2.py files are for 
creating clean training and test set csv files for using with the tree-based 
models. This creates 2 versions: one is the original trimmed data set, and 
then the "v2" adds in more features to see what happens.

tree_model.py is the function to run the regression. To run:

>> python tree_model.py <model> <version>

where <model> can be any of: randomforest, gradboost, or xgboost
This is the regrssion model, telling the script to run the corresponding 
sklearn regressor with a GridSearchCV

and <version> is either 'v1' or 'v2'
This is the version of the train/test sets to use: v1 corresponds to the 
output of feature_engineering.py and v2 corresponds to 
feature_engineering_v2.py

example: >> python tree_model.py gradboost v2

