#!/usr/bin/env python
# coding: utf-8

# # Feature engineering
# In feature engineering, we carry out feature engineering, extract new features that are relevant for the problem. For Spotify data set, three additional features are extracted from Spotify API and Genius API. Two scripts are developed for the data extraction.
# 
# 1. Extract artist genre using Spotify API (JAVA) [Here](https://github.com/MacyChan/spotify-user-behaviour-predictor/blob/master/spotify_user_behaviour_predictor/scr/getLyrics.py)
# 2. Extract song lyric using self developed package (Python) [Here](https://github.com/UBC-MDS/pylyrics)

# In[1]:


import pandas as pd
import re


# ## Reading the data CSV
# Read in the data CSV and store it as a pandas dataframe named `spotify_df`.

# In[2]:


spotify_df = pd.read_csv('data/spotify_data.csv', index_col = 0 )
spotify_df.head(6)


# ## Artist Information
# `genres` and `popularity` are extracted from Spotify API, which included the genres and popularity of the corresponding artist.

# In[3]:


artist_df = pd.read_csv('data/artist_info.csv', index_col = 0 )
artist_df.head(6)


# Pivot the artist table with `genres` in columns and `artist` in row, count the number of `artist` appeared.

# In[4]:


artist_df_pivot = (
    artist_df.pivot_table(
        index="name",
        columns="genres",
        values="popularity",
        #aggfunc=lambda x: len(x.unique()),
        aggfunc="count",
    )
    .add_prefix("genres_")
    .reset_index()
)

artist_df_pivot.fillna(0, inplace=True)


# Join pivoted artist table to original table

# In[5]:


spotify_df = spotify_df.merge(artist_df_pivot, left_on='artist', right_on='name')
spotify_df = spotify_df.drop(['name'], axis=1)
spotify_df.head(6)


# ## Song Information
# `lyrics` is extracted from Genius API, which included the lyrics of the corresponding song.  
# A python script is developed for scraping the lyrics [here]()

# In[6]:


lyrics_df = pd.read_csv('data/lyrics_info.csv', index_col = 0 )
lyrics_df.head(6)


# Join the lyrics with the dataframe.

# In[7]:


spotify_df = spotify_df.merge(lyrics_df)
spotify_df.head(6)


# ## Export CSV
# Export new csv with additional feature for further machine learning process.

# In[8]:


spotify_df.to_csv('data/spotify_df_processed.csv',index=False)

