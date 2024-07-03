#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
from pyDOE2 import *


# In[2]:


dim=80                      #需要生成数据的维度
num_data=160                #需要生成数据的数量
lb=-3                       #数据的下界
ub=3                        #数据的上界


# In[3]:


data=lhs(dim,samples=num_data,criterion='maximin')   
data=data*(ub-lb)+lb


# In[10]:


with open("data.csv",'w',newline='') as t: 
    writer=csv.writer(t)                                
    writer.writerows(data)      


# In[ ]:




