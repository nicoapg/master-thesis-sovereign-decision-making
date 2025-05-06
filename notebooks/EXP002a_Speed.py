#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("..")


# # *dummy_p100_c10000*

# In[2]:


get_ipython().run_line_magic('run', 'main_benchmarking.py      --dataset_name dummy --n_principles 100 --n_comparisons 10000      --benchmarks fitting_duration      --learnings offline      --show_figs')


# In[ ]:




