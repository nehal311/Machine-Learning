#!/usr/bin/env python
# coding: utf-8

# In[1]:


#installing scikit-learn package
get_ipython().system('pip install -U scikit-learn')


# In[2]:


import pandas as pd


# In[3]:


#to read training dataset:
#1. Converted the text file into csv file by saving text file as .csv file
#2. using pandas read_csv function read the csv file by passing the path of the file. 
#   For better understanding separated each value by tab and added names to each column.
dataset = pd.read_csv('wildfires_training.csv',sep = '\t',  
                    names =['fire','year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed'
])


# In[4]:


dataset


# In[5]:


# Data Preprocessing : As Column Fire contains string value and other contains numeric value, Machine Learning model cant handle text 
# hence converted into string. Before that checked unique values of fire column, it has 7 unique value such as ['no   ', 'yes   ', 'yes', 'yes ', 'no', 'no ', 'no    '],
# Originally, there are only 2 unique values are present "yes" and "no" but due to extra space in start and end it becomes unique value.
# To get the clean data ,cut the extra space using str.strip().
dataset.fire = dataset.fire.str.strip()


# In[6]:


dataset.fire.unique()


# In[7]:


dummies = pd.get_dummies(dataset.fire)
dummies


# In[8]:


y_train= dataset.fire
y_train


# In[9]:


dataset= pd.concat([dataset,dummies],axis = 'columns')
dataset


# In[10]:


dataset.drop('fire',axis = 'columns',inplace = True)
dataset


# In[11]:


#to check if any NA value is there in the data set
dataset.columns[dataset.isna().any()]


# In[12]:


len(dataset)


# In[13]:


from sklearn.naive_bayes import GaussianNB
model= GaussianNB()


# In[15]:


model.fit(dataset,y_train)


# In[16]:


model.score(dataset,y_train)

