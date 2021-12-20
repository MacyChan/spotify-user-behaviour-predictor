#!/usr/bin/env python
# coding: utf-8

# # Statistic Inference
# - This is a R base notebook  
# 
# We saw some pattern in EDA, naturally, we would like to see if the different between feature are significantly related to the target.

# ## Import libaries

# In[1]:


library(tidyverse)
library(broom)
library(GGally)


# ## Reading the data CSV
# Read in the data CSV and store it as a pandas dataframe named `spotify_df`. 

# In[2]:


spotify_df <- read_csv("data/spotify_data.csv")
head(spotify_df)


# ## Regression

# ### Data Wrangle
# - Remove `song_title` and `artist` for relationship study by regression. As both of them are neither numerical nor categorical features.

# In[3]:


spotify_df_num <- spotify_df[2:15]
head(spotify_df_num)


# ## Set up regression model

# Here, I am interested in determining factors associated with `target`. In particular, I will use a Multiple Linear Regression (MLR) Model to study the relation between `target` and all other features.

# In[4]:


ML_reg <- lm( target ~ ., data = spotify_df_num) |> tidy(conf.int = TRUE)

ML_reg<- ML_reg |>
    mutate(Significant = p.value < 0.05)

ML_reg


# - We can see that a lot of features are statiscally correlated with target. They are listed in the table below.

# In[5]:


ML_reg |>
    filter(Significant == TRUE) |>
    select(term) 


# ### GGpairs
# Below is the ggpair plots to visual the correlation between different features.

# In[6]:


ggpairs(data = spotify_df_num)

