#!/usr/bin/env python
# coding: utf-8

# # Statistic Inference
# - This is a Python base notebook
# - Using `rpy2` for R functions
# 
# We saw some pattern in EDA, naturally, we would like to see if the different between feature are significantly related to the target.

# ## Import libaries

# In[1]:


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


# In[2]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[3]:


get_ipython().run_cell_magic('R', '', 'library(tidyverse)\nlibrary(broom)\nlibrary(GGally)')


# ## Reading the data CSV
# Read in the data CSV and store it as a pandas dataframe named `spotify_df`. 

# In[4]:


get_ipython().run_cell_magic('R', '', 'spotify_df <- read_csv("data/spotify_data.csv")\nhead(spotify_df)')


# ## Regression

# ### Data Wrangle
# - Remove `song_title` and `artist` for relationship study by regression. As both of them are neither numerical nor categorical features.

# In[5]:


get_ipython().run_cell_magic('R', '', 'spotify_df_num <- spotify_df[2:15]\nhead(spotify_df_num)')


# ## Set up regression model

# Here, I am interested in determining factors associated with `target`. In particular, I will use a Multiple Linear Regression (MLR) Model to study the relation between `target` and all other features.

# In[6]:


get_ipython().run_cell_magic('R', '', 'ML_reg <- lm( target ~ ., data = spotify_df_num) |> tidy(conf.int = TRUE)\n\nML_reg<- ML_reg |>\n    mutate(Significant = p.value < 0.05) |>\n    mutate_if(is.numeric, round, 3)\n\nML_reg')


# - We can see that a lot of features are statiscally correlated with target. They are listed in the table below.

# In[7]:


get_ipython().run_cell_magic('R', '', 'ML_reg |>\n    filter(Significant == TRUE) |>\n    select(term) ')


# ### GGpairs
# Below is the ggpair plots to visual the correlation between different features.

# In[8]:


get_ipython().run_cell_magic('R', '', 'ggpairs(data = spotify_df_num)')

