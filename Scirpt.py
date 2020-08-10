# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data Visualization Library
import seaborn as sns #Data Visualization Library


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
#Loading the Data
sample_sub = pd.read_csv('../input/neolen-house-price-prediction/sample_submission.csv')
print(sample_sub.info())
print(sample_sub.head())

# %% [code]
X = pd.read_csv('../input/neolen-house-price-prediction/train.csv',index_col = 'Id')
y = pd.read_csv('../input/neolen-house-price-prediction/test.csv',index_col = 'Id')
print(X.info())
print(X.head())
print(y.info())
print(y.head())

# %% [code]
#Looking Into the Data
print('Our training dataset has {} rows and {} columns'.format(X.shape[0],X.shape[1]))
print('Our Test data has {} rows and {} columns'.format(y.shape[0],y.shape[1]))

# %% [code]
#Data Analysis

#Expensiveness of Houses
print("The cheapest house sold for ${:,.0f} and the most expensive house sold for ${:,.0f}".format(X.SalePrice.min(),X.SalePrice.max()))
print("The Average Sales Price is ${:,.0f}, while median is ${:,.0f}".format(X.SalePrice.mean(),X.SalePrice.median()))

# %% [markdown]
# ****General Questions About Dataset

# %% [code]
X.SalePrice.hist(bins=75,rwidth = .8,figsize=(14,4))
plt.title('House Prices')
plt.show()

# %% [markdown]
# From the Above Graph we can say the prices ranges from 50k to 250k. Mostly houses are in the range of 100k to 200k.

# %% [code]
#Analysing the Year of Built
print('Oldest house built in the Year {} and Newest house built in the Year {}'.format(X.YearBuilt.min(),X.YearBuilt.max()))

# %% [code]
X.YearBuilt.hist(bins=14,rwidth=.9,figsize=(12,4))
plt.title('When were the Houses built?')
plt.show()

# %% [markdown]
# Mostly Houses were sold after the year 1950

# %% [code]
X.groupby(['YrSold','MoSold']).count().plot(kind='barh',figsize=(14,21.8))
plt.title('Year Of House Selling')
plt.show()

# %% [markdown]
# Mostly the Sale of Houses are done in the month of June-July

# %% [code]
X.groupby('Neighborhood').count().plot(kind='barh',figsize=(4,21.85))
plt.title('What Neighbourhoods are houses in?')
plt.show()

# %% [markdown]
# This is a complete example of how Neighborhood plays a key role in the buying of the house.There is more sale of houses in the neoghborhood of North Ames,Old Town and very less sale in Bluestem,Northpark Villa.

# %% [code]
sns.pairplot(X[["SalePrice", "LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]], diag_kind="kde")

# %% [code]
#Removing SalesPrice from X
Y = X.SalePrice #notice the capital y
X = X.drop(columns=['SalePrice'],axis=1)
Y

# %% [code]
X

# %% [markdown]
# Find numerical and Categorical Columns

# %% [code]
for col in X.columns:
    print(col,X[col].dtype)
print('*******************************************')
print('*******************************************')
for col in y.columns:
    print(col,y[col].dtype)

# %% [code]
#Changing Data to Numericals
X_num = X.select_dtypes(exclude=['object'])
X_num

# %% [code]
X_num.isnull().sum()

# %% [code]
y_num = y.select_dtypes(exclude=['object'])
y_num

# %% [code]
y_num.isnull().sum()

# %% [markdown]
# **Impution Process**
# The Imputer fills missing values with some statistics (e.g. mean, median, ...) of the data. To avoid data leakage during cross-validation, it computes the statistic on the train data during the fit, stores it and uses it on the test data, during the transform.

# %% [code]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X_num))
X_imputed.columns = X_num.columns

y_imputed = pd.DataFrame(imputer.transform(y_num))
y_imputed.columns = y_num.columns

# %% [code]
X_imputed

# %% [code]
X_imputed.isnull().sum()

# %% [code]
y_num

# %% [code]
y_imputed.isnull().sum()

# %% [markdown]
# **Boosting**
# Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher. Note that a weak learner is one which is slightly better than random guessing. For example, a decision tree whose predictions are slightly better than 50%.

# %% [code]
parameters = {
    'n_estimators':list(range(100,1001,100)),
    'learning_rate':[x/100 for x in range(5,100,10)],
                    'max_depth':list(range(6,90,10))
    }
parameters

# %% [code]
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
gsearch = GridSearchCV(XGBRegressor(random_state=1),
                       param_grid = parameters, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=4,cv=5, verbose=1)


# %% [code]

gsearch.fit(X_imputed,Y)


# %% [code]
best_n_estimators = gsearch.best_params_.get('n_estimators')
best_n_estimators

# %% [code]
best_learning_rate = gsearch.best_params_.get('learning_rate')
best_learning_rate

# %% [code]
best_max_depth = gsearch.best_params_.get('max_depth')
best_max_depth

# %% [code]
final_model = XGBRegressor(n_estimators=best_n_estimators,random_state=1,learning_rate = best_learning_rate,max_depth=best_max_depth)
final_model.fit(X_imputed,Y)

# %% [code]
preds_test = final_model.predict(y_imputed)
print(len(preds_test))
print(final_model.predict(y_imputed.head(10)))
print(len(X_imputed.index))

# %% [code]
output = pd.DataFrame({'Id':y.index,
                      'SalePrice':preds_test})
print(output)
output.to_csv('submission.csv',index=False)
print('done')
