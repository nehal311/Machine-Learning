#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


# In[2]:


dataset = pd.read_csv('galway_rental.csv',sep = '\t')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


sns.pairplot(dataset)


# In[7]:


sns.heatmap(dataset.corr(), annot=True)


# In[8]:


categorical_columns = ['type','heating']
dataset = pd.get_dummies(dataset, columns = categorical_columns)
dataset


# In[9]:


dataset.ber.unique() #array(['exempt', 'b3', 'c1', 'd1', 'd2', 'b1', 'c2', 'c3', 'b2', 'a','e2', 'e1', 'f', 'g'], dtype=object)
ber = ['a', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'e1', 'e2', 'f', 'g', 'exempt']
ordi = OrdinalEncoder(categories = [ber])
ordi.fit(dataset[['ber']])
dataset.ber =ordi.transform(dataset[['ber']]) #a =1, b1=2, b2=2, b3=3, c1=4, c2=5, c3=6, d1=7, d2=8, e1=9, e2=10, f=11,g=12, exempt=13


# In[10]:


dataset.balcony.unique() #array(['no', 'yes'], dtype=object)
balcony = ['yes','no']
ordi = OrdinalEncoder(categories = [balcony])
ordi.fit(dataset[['balcony']])
dataset.balcony =ordi.transform(dataset[['balcony']])


# In[11]:


dataset.floor.unique() #array(['ground', 'second', 'first', 'third'], dtype=object)
floor = ['ground', 'second', 'first', 'third']
ordi = OrdinalEncoder(categories = [floor])
ordi.fit(dataset[['floor']])
dataset.floor =ordi.transform(dataset[['floor']]) #ground=0, first=1, second=2, third=3


# In[12]:


dataset


# In[13]:


from sklearn.model_selection import train_test_split
train_data, test_data =  train_test_split(dataset, test_size=0.25,random_state=42) 


# In[14]:


train_data


# In[15]:


train_label = train_data["price_per_month"]
train_data  = train_data.drop("price_per_month", axis=1)


# In[16]:


test_label = test_data["price_per_month"]
test_data  = test_data.drop("price_per_month", axis=1)


# In[17]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(train_data)
train_data = scalar.fit_transform(train_data)
test_data  = scalar.transform(test_data)


# In[26]:


from sklearn.linear_model import LinearRegression
linearReg = LinearRegression()
linearReg.fit(train_data, train_label)
print(len(train_data))
print(len(train_label))


# In[27]:


predictions = linearReg.predict(test_data)
len(test_data)


# In[20]:


plt.scatter(test_label,predictions)


# In[21]:


from sklearn.metrics import mean_squared_error
from sklearn import metrics as m 
MSE = mean_squared_error(test_label,predictions)
RMSE = np.sqrt(MSE)
#R2_Score = linearReg.score(test_label,predictions)
r2 = m.r2_score(test_label,predictions)


# In[22]:


print("The Mean Squared Error :", MSE)
print("The Root-mean-square deviation: ",RMSE)
print("The R2 Score:",r2*100)


# In[23]:


from sklearn.neighbors import KNeighborsRegressor
KNNModel = KNeighborsRegressor(n_neighbors =5)
KNNModel.fit(train_data, train_label)
KNN_predictions = np.round(KNNModel.predict(test_data),0)


# In[24]:


MSE = mean_squared_error(test_label,KNN_predictions)
RMSE = np.sqrt(MSE)
#R2_Score = linearReg.score(test_label,predictions)
r2 = m.r2_score(test_label,KNN_predictions)


# In[25]:


print("The Mean Squared Error :", MSE)
print("The Root-mean-square deviation: ",RMSE)
print("The R2 Score:",r2 *100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




