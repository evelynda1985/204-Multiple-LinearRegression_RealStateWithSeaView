#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression with Dummies - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year_view.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. 
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size', 'year', and 'view'.
# 
# #### Regarding the 'view' variable:
# There are two options: 'Sea view' and 'No sea view'. You are expected to create a dummy variable for view and include it in the regression
# 
# Good luck!

# ## Import the relevant libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sn
sn.set()


# ## Load the data

# In[5]:


data = pd.read_csv('real_estate_price_size_year_view.csv')


# In[34]:


data.head()


# In[38]:


data.describe(include='all')


# In[35]:


data_copy = data.copy()


# ## Create a dummy variable for 'view'

# In[41]:


data_copy['view'] = data_copy['view'].map({'Sea view':1, 'No sea view':0})


# In[42]:


data_copy


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[43]:


y = data_copy['price']
x1 = data_copy[['size','year','view']]


# ### Regression

# In[44]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[ ]:




