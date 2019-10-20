#!/usr/bin/env python
# coding: utf-8

# In[95]:


import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split


# In[96]:


# path for the csv files
DATA_PATH = os.path.join(os.getcwd(), 'data')

# loading data to pandas dataframe
def load_data(file_name): 
    file_path = os.path.join(DATA_PATH, file_name) 
    return pd.read_csv(file_path, parse_dates = ['purchase_date', 'release_date'])


# In[97]:


def extract_dateinfo(df, col_name):
    df[col_name+'_year'] = df.loc[:,col_name].apply(lambda x: x.year)
    df[col_name+'_month'] = df.loc[:,col_name].apply(lambda x: x.month)
    df[col_name+'_day'] = df.loc[:,col_name].apply(lambda x: x.day)
    
    return df


# In[98]:


#Outlier detection
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    
    outlier_indices = []
    
    for col in features:
        # calculating interquartile range
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1
        print(IQR)
        
        outlier_step = 1.5 * IQR
        
        
        
        # get the indices of outliers for feature col
        outliers_in_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        # append the indices to oulier_indices
        outlier_indices.extend(outliers_in_col)
    
    outlier_indices = Counter(outlier_indices)
    result = list(k for k, v in outlier_indices.items() if v > n)
    
    return result


# In[99]:


train_set = load_data('train.csv')
test_set = load_data('test.csv')



# In[100]:


outliers_to_drop = detect_outliers(train_set, 2 ,['price', 'total_positive_reviews', 'total_negative_reviews'])
train_set.loc[outliers_to_drop]
train_set = train_set.drop(outliers_to_drop, axis = 0).reset_index(drop=True)

train_len = train_set.shape[0]
test_len = test_set.shape[0]


# In[101]:


# game_info.head()
game_info =  pd.concat(objs=[train_set, test_set], axis=0, sort=False).reset_index(drop=True)
game_info


# In[102]:


# check null values
game_info.fillna(np.nan, inplace=True)

#fill missing purchase date with the most frequent value in purchase_date column
game_info['purchase_date'].fillna(game_info['purchase_date'].mode()[0], inplace=True)

#fille missing number of positive_reviews and negative_reviews with zeros
game_info['total_positive_reviews'].fillna(0.0, inplace=True)
game_info['total_negative_reviews'].fillna(0.0, inplace=True)

#transfer boolean values to 1(true) and 0(false)
game_info['is_free'] = game_info['is_free'].map({False: 0.0, True: 1.0})

#drop outliers


# split strings in the categorical columns
game_info['genres'] = game_info['genres'].str.split(',')
game_info['categories'] = game_info['categories'].str.split(',')
game_info['tags'] = game_info['tags'].str.split(',')

game_info

#game_info.isnull().sum()


# In[103]:


#dataframe with only categorical values
cate_df = game_info[['genres', 'categories', 'tags']]


# one_hot encoding categorical columns 
genres_df = pd.get_dummies(cate_df['genres'].apply(pd.Series).stack()).sum(level=0)
categories_df = pd.get_dummies(cate_df['categories'].apply(pd.Series).stack()).sum(level=0)
tags_df = pd.get_dummies(cate_df['tags'].apply(pd.Series).stack()).sum(level=0)

# contatenate categorical dataframes with game_info
game_df = pd.concat([game_info.drop(columns=['genres', 'categories', 'tags']), genres_df, categories_df, tags_df], axis=1, sort=False)
game_df


# In[104]:


# drop duplicate columns
game_df = game_df.loc[:, ~game_df.columns.duplicated()]
game_df


# In[105]:


game_df = extract_dateinfo(game_df, 'purchase_date')
game_df = extract_dateinfo(game_df, 'release_date')
game_df.drop(columns=['purchase_date', 'release_date'], inplace=True)
#game_df.drop(['purchase_date'])
#'purchase_date' in game_df
#game_df.loc[:,'purchase_date']
game_df


# In[108]:


train_data = game_df[:train_len]
test_data = game_df[train_len:]

train_label = train_data['playtime_forever']
train_data = train_data.drop(columns=['playtime_forever'])

# split game_info into training and validating datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


test_data.drop(columns=['playtime_forever'],inplace=True)

print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('y_train shape : ', y_train.shape)
print('y_val shape : ', y_val.shape)


# In[110]:


#lab_enc = preprocessing.LabelEncoder()
#train_encoded = lab_enc.fit_transform(y_train)
#val_encoded = lab_enc.fit_transform(y_val)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_val)

y_test_pred = linreg.predict(test_data)

print("Mean squared error: %.2f"
      % mean_squared_error(y_val, y_pred))

print(y_test_pred)


# In[119]:


import csv

result_df = pd.DataFrame(y_test_pred, columns =['playtime_forever']) 
result_df.index.name = 'id'
result_df.to_csv('result.csv')


# In[ ]:




