#!/usr/bin/env python
# coding: utf-8

# # Conclusion
# 
# - This is a Python base notebook  

# ## Summary of the whole project

# In[1]:


import pandas as pd
results = pd.read_csv('data/model_results.csv', index_col = 0 )
results.reset_index().rename(index= {0: 'fit time', 1: 'score time', 2: 'test accuracy', 3: 'train accuracy', 4: 'test ROC AUC', 5: 'train ROC AUC'})


# Amongst the models, `LGBMClassifier` is the best model. Even though `CatBoostClassifier` has the best accuracy and ROC AUC score, the fit time for`cat_boost` is slow. It can be a concern as the algorithm is likely to refit every user by the latest song listening history whenever the user wants to update their playlist. For `LGBMClassifier`, the test accuracy is **0.950**, which is 0.004 lower than `CatBoostClassifier`, but the fit time is 20 times shorter. 
# 
# Therefore, `LGBMClassifier` will be used for the Soptify user behavior prediction.
