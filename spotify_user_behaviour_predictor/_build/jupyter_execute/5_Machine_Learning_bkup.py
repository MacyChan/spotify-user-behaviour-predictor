#!/usr/bin/env python
# coding: utf-8

# # Machine Learning analysis
# 
# - This is a Python base notebook  
# 
# Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset contains a number of features of songs from 2017 and a binary variable `target` that represents whether the user liked the song (encoded as 1) or not (encoded as 0). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). 

# ## Imports

# In[1]:


import pandas as pd
from sklearn.model_selection import (train_test_split)


# ### Reading the data CSV and Spilt the data
# Read in the data CSV and store it as a pandas dataframe named `spotify_df`.

# In[2]:


spotify_df = pd.read_csv('data/spotify_data.csv', index_col = 0 )
spotify_df.head(6)


# In[5]:


train_df, test_df = train_test_split(spotify_df, test_size=0.2, random_state=123)


# <br><br>

# ## Model building
# Remove `song_title`, separate data to `X_train`, `y_train`, `X_test`, `y_test`.

# In[6]:


X_train, y_train = train_df.drop(columns=["song_title", "target"]), train_df["target"]
X_test, y_test = test_df.drop(columns=["song_title", "target"]), test_df["target"]

