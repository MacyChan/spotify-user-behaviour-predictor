#!/usr/bin/env python
# coding: utf-8

# # Conclusion
# 
# - This is a Python base notebook  

# ## Summary of the whole project

# In[1]:


import pandas as pd
results = pd.read_csv('data/model_results.csv', index_col = 0 )
results


# Amongst the models, it is clear that `CatBoostClassifier` has the best accuracy and ROC AUC score. Moreover, the difference in scores between the test set and validation set are comparatively small, suggesting that there is minimal optimisation bias. Based on our test scores, our model is able to identify about **77.2%** of the cases.
# 
# One thing that needs to be kept in mind is that, the fit time for`cat_boost` is slow. It can be a concern as the algorithm is likely to refit every user by the latest song listening history whenever the user wants to update their playlist. Moreover, the score time of `Cat_Boost` is the slowest among all models but acceptable. The processing time (fit + score time) is critical for the application.  
# 
# Hence, the other options after `Cat_Boost` may be `RandomForest` / `LGBM`, which require less processing time but similar scoring result. Further model training and new training set is needed for continuous model studying.
