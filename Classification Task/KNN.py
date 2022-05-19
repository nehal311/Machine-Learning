#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U scikit-learn')


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


dataset_train = pd.read_csv('wildfires_training.csv',sep = '\t',  
                    names =['fire','year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed'
])
dataset_train


# In[4]:


dataset_test = pd.read_csv('wildfires_test.csv',sep = '\t',  
                    names =['fire','year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed'
])
dataset_test


# In[5]:


y_test = dataset_test.fire
y_test.unique()


# In[9]:


dummiestest = pd.get_dummies(dataset_test.fire)
dummiestest


# In[10]:


y_test_yeswithsinglespace= dummiestest['yes ']
y_test_yeswithsinglespace


# In[11]:


y_train=dataset_train.fire
y_train
y_train.unique()


# In[12]:


dummies = pd.get_dummies(dataset_train.fire)
dummies


# In[13]:


#dataset_test
dataset_test.drop('fire',axis = 'columns',inplace = True)
dataset_test


# In[14]:


y_train_yeswithsinglespace = dummies['yes']
y_train_yeswithsinglespace.value_counts


# In[15]:


dataset_train.drop('fire',axis = 'columns',inplace = True)
dataset_train


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(dataset_train,y_train_yeswithsinglespace)


# In[17]:


classifier.score(dataset_train,y_train_yeswithsinglespace)


# In[18]:


y_pred = classifier.predict(dataset_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(dataset_test,y_pred))
print(confusion_matrix(dataset_test,y_pred))


# In[ ]:


len(dataset_train)


# In[ ]:





# In[ ]:





# In[ ]:




