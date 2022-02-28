#!/usr/bin/env python
# coding: utf-8

# # Appendix
# 
# - This is a Python base notebook  

# ## Demo of scrapping lyrics for the first 10 songs using pyLyrics2

# In[1]:


from pylyrics2 import extract_lyrics as pl
from pylyrics2 import clean_text as ct
import pandas as pd
import re
import sys, os


# In[2]:


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# In[3]:


spotify_df = pd.read_csv('data/spotify_data.csv', index_col = 0 )


# In[4]:


spotify_df[["song_title", "artist"]].head()


# In[5]:


def scrapLyrics(song_title, artist):
    with HiddenPrints():
        try:
            return pl.extract_lyrics(song_title, artist)
        except:
            return " "


# In[6]:


lyrics = []
for i in range(10):  # range(len(spotify_df)):
    song_title = ct.clean_text(text=spotify_df.iloc[i]["song_title"], bool_contra_dict=False)
    artist = ct.clean_text(text=spotify_df.iloc[i]["artist"], bool_contra_dict=False)
    lyric = scrapLyrics(song_title, artist)
    if lyric.strip() != "":
        clean_lyric = ct.clean_text(lyric)
    else:
        clean_lyric = ""
    lyrics.append([song_title, artist, clean_lyric])


# In[7]:


pd.DataFrame(lyrics, columns = ["song_title", "artist", "lyrics"]).head(10)

