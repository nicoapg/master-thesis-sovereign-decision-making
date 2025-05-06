#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("..")


# # *uniform_absolute_p25_c2500*

# In[2]:


get_ipython().run_line_magic('run', 'main_benchmarking.py      --dataset_name synthetic      --n_principles 25 --n_comparisons 2500      --sampling uniform --voting absolute      --benchmarks converging_paces --l_batches 100      --show_figs')


# # *uniform_noisy_p25_c2500_noise25*

# In[3]:


get_ipython().run_line_magic('run', 'main_benchmarking.py      --dataset_name synthetic      --n_principles 25 --n_comparisons 2500      --sampling uniform --voting noisy --noise 0.25      --benchmarks converging_paces --l_batches 100      --show_figs')


# In[ ]:




