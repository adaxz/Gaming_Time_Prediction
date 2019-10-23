#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import pandas as pd
import numpy as np
from collections import Counter
#from sklearn.pipeline import make_pipeline, Pipeline

#from sklearn import preprocessing
from sklearn import utils

import datetime
import csv
#import nltk
from sklearn.preprocessing import MultiLabelBinarizer


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split


# In[73]:


# path for the csv files
DATA_PATH = os.path.join(os.getcwd(), 'data')

# loading data to pandas dataframe
def load_data(file_name): 
    file_path = os.path.join(DATA_PATH, file_name) 
    return pd.read_csv(file_path, parse_dates = ['purchase_date', 'release_date'])

def extract_dateinfo(df, col_name):
    year = df[col_name].dt.year
    #month = df[col_name].dt.month
    #day = df[col_name].dt.day
    
    df.loc[:, col_name+'_year'] = year
    #df.loc[:, col_name+'_month'] = month
    #df.loc[:, col_name+'_day'] = day
    return df

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
        
        outlier_step = 1.5 * IQR  
        
        # get the indices of outliers for feature col
        outliers_in_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        # append the indices to oulier_indices
        outlier_indices.extend(outliers_in_col)
    
    outlier_indices = Counter(outlier_indices)
    result = list(k for k, v in outlier_indices.items() if v > n)
    
    return result


# In[74]:


train_set = load_data('train.csv')
test_set = load_data('test.csv')

# Drop outliers from training data
outliers_to_drop = detect_outliers(train_set, 2 ,['price', 'total_positive_reviews', 'total_negative_reviews'])
train_set.loc[outliers_to_drop]
train_set = train_set.drop(outliers_to_drop, axis = 0).reset_index(drop=True)

train_len = train_set.shape[0]
test_len = test_set.shape[0]

# merge training data and testing data 
game_info =  pd.concat(objs=[train_set, test_set], axis=0, sort=False).reset_index(drop=True)
#game_info.drop(columns=['id'], inplace=True)


# check null values
game_info.fillna(np.nan, inplace=True)

#fill missing purchase date with the most frequent value in purchase_date column
game_info['purchase_date'].fillna(game_info['purchase_date'].mode()[0], inplace=True)

#fille missing number of positive_reviews and negative_reviews with zeros
game_info['total_positive_reviews'].fillna(0.0, inplace=True)
game_info['total_negative_reviews'].fillna(0.0, inplace=True)

#transfer boolean values to 1(true) and 0(false)
game_info['is_free'] = game_info['is_free'].map({False: 0.0, True: 1.0})

# extract year value
game_info = extract_dateinfo(game_info, 'purchase_date')
game_info = extract_dateinfo(game_info, 'release_date')
game_info.drop(columns=['purchase_date', 'release_date'], inplace=True)


# In[75]:


# split strings in the categorical columns
game_info['genres'] = game_info['genres'].str.split(',').apply(lambda x: list(map(lambda y: y.lower(), x)))
game_info['categories'] = game_info['categories'].str.split(',').apply(lambda x: list(map(lambda y: y.lower(), x)))
game_info['tags'] = game_info['tags'].str.split(',').apply(lambda x: list(map(lambda y: y.lower(), x)))


#dataframe with only categorical values
cate_df = game_info[['genres', 'categories', 'tags']]

#one_and_two = set(genres) + set(categories) - set(genres) 
#unique_tags = one_and_two + set(tags) - one_and_two 

def _unique_tags_(row):
    col1 = row.iloc[0]
    col2 = row.iloc[1]
    col3 = row.iloc[2]
    one_two = set(col1) | set(col2)
    return list(one_two | set(col3))

game_info.loc[:, 'all_cate'] = cate_df.apply(_unique_tags_, axis=1)
game_info.loc[:, 'cate_count'] = game_info['all_cate'].apply(lambda x: len(x))
game_info.drop(columns=['genres', 'categories', 'tags'], inplace=True)

#cate_df.loc[:, 'genres_count'] = cate_df['genres'].apply(lambda x: len(x))
#cate_df.loc[:, 'cates_count'] = cate_df['categories'].apply(lambda x: len(x))
#cate_df.loc[:, 'tags_count'] = cate_df['tags'].apply(lambda x: len(x))


# In[76]:


game_info.drop(columns=['id'], inplace=True)


# In[77]:


# Create MultiLabelBinarizer object
mlb = MultiLabelBinarizer()

# One-hot encode data
cate_one_hot = mlb.fit_transform(game_info['all_cate'])

#how many games belong to each category
cate_count = list(np.count_nonzero(matrix, axis=0))
# list of all the categories
cates = list(one_hot.classes_)

cate_dic = dict(zip(cates, cate_count))


# In[78]:


cate_dic


# In[79]:


game_info.drop(columns=['all_cate'], inplace=True)
#game_info.iloc[0, 7]


# In[80]:


# one_hot encoding categorical columns 
#genres_df = pd.get_dummies(cate_df['genres'].apply(pd.Series).stack()).sum(level=0)
#categories_df = pd.get_dummies(cate_df['categories'].apply(pd.Series).stack()).sum(level=0)
#tags_df = pd.get_dummies(cate_df['tags'].apply(pd.Series).stack()).sum(level=0)

# contatenate categorical dataframes with game_info
#game_df = pd.concat([game_info.drop(columns=['genres', 'categories', 'tags']), categories_df, tags_df], axis=1, sort=False)
#game_df = pd.concat([game_df, cate_df.loc[:, ['genres_count','cates_count', 'tags_count']]], axis=1, sort=False)
#game_df


# In[81]:


# drop duplicate columns
# game_df = game_df.loc[:, ~game_df.columns.duplicated()]
# genres = list(game_df.columns[6:349])

# print(len(genres))
# with open('genres.txt', 'w') as f:
#     for item in genres:
#         f.write("%s\n" % item)


# In[82]:


train_data = game_info[:train_len]
test_data = game_info[train_len:]


# In[83]:


plt.figure(figsize=(5, 9))

features = ['playtime_forever', 'total_positive_reviews', 'purchase_date_year', 'release_date_year','cate_count', 'price']
g = sns.heatmap(train_data[features].corr(),annot=True, cmap = "coolwarm")


# In[84]:


train_label = train_data['playtime_forever']
train_data = train_data.drop(columns=['playtime_forever'])
test_data.drop(columns=['playtime_forever'],inplace=True)


# print('X_train shape: ', X_train.shape)
# print('X_val shape: ', X_val.shape)
# print('y_train shape : ', y_train.shape)
# print('y_val shape : ', y_val.shape)


# In[85]:


train_data.head()


# In[86]:


X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=42)


# In[96]:


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)

def rmse(y_val, y_val_pred):
    return np.sqrt(mean_squared_error(y_val, y_val_pred))
score = make_scorer(rmse, greater_is_better=True)
#linreg = LinearRegression()
#linreg.fit(X_train, y_train)

model = linear_model.LinearRegression()
y_val_pred = linreg.predict(X_val)
y_test_pred = linreg.predict(test_data)
print(cross_val_score(model, X_train, y_train, cv=3, scoring=score))  
#print(linreg.score(X_train, y_train))

print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_val, y_val_pred)))


# In[35]:




result_df = pd.DataFrame(y_test_pred, columns =['playtime_forever']) 
result_df.index.name = 'id'
result_df.to_csv('result.csv')


# In[ ]:




