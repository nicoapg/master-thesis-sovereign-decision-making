#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("..")


# # *experiment001*

# In[2]:


get_ipython().run_line_magic('run', 'main_processing.py --dataset experiment001')
get_ipython().run_line_magic('run', 'main_benchmarking.py --dataset_name experiment001 --show_figs')


# # *experiment002*

# In[3]:


get_ipython().run_line_magic('run', 'main_processing.py --dataset experiment002')
get_ipython().run_line_magic('run', 'main_benchmarking.py --dataset_name experiment002 --show_figs')


# In[ ]:




