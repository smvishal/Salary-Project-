#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Read csv file and convert into dataframe

df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/Salaries.csv')
df.head(5)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.sample(n=20)


# In[8]:


df.columns


# In[9]:


# Checking The NullValues
df.isnull().sum()  


# In[10]:


from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
df2 = lab_enc.fit_transform(df['rank'])
pd.DataFrame(df2)


# In[11]:


df['rank'] = df2
df


# In[12]:


df2 = lab_enc.fit_transform(df['sex'])
pd.DataFrame(df2)


# In[13]:


df['sex'] = df2
df


# In[14]:


df2 = lab_enc.fit_transform(df['discipline'])
pd.DataFrame(df2)


# In[15]:


df['discipline'] = df2
df


# In[16]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


# Let's see how data is distributed for every column

plt.figure(figsize=(15,10), facecolor='pink')
plotnumber = 1

for column in df:
    if plotnumber<=6:
        ax = plt.subplot(2,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize=15)
        
    plotnumber+=1
plt.tight_layout() 


# In[19]:


# Divide dataset into features and label
y = df['salary']
X = df.drop(columns = ['salary'])
y


# In[20]:


X


# In[21]:


# Visualizing relationship

plt.figure(figsize=(12,8), facecolor = 'yellow')
plotnumber = 1

for column in X:
    if plotnumber<=6:
        ax = plt.subplot(2,3,plotnumber)
        plt.scatter(X[column],y)
        plt.xlabel(column,fontsize=10)
        plt.ylabel('salary',fontsize=10)
        
    plotnumber+=1
plt.tight_layout()  


# In[22]:


# Data Scaling. Formula Z = (X - mean)/ std

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[23]:


X_scaled


# In[24]:


# Split data into train and test. Model will be built on training data and tested on test data.
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state = 356)
y_train.head()


# In[25]:


regression = LinearRegression()

regression.fit(x_train, y_train)


# In[26]:


#Predict the salary on given features
df.head(5)


# In[27]:


# Since we have already fit the scaler, you can transform the data

print ('Salary is: ', regression.predict(scaler.transform([[2, 1, 45, 39, 1]])))


# In[28]:


# Adjusted R2 score
regression.score(x_train, y_train)


# In[29]:


y_pred = regression.predict(x_test)

y_pred


# In[30]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted model')
plt.show()


# In[31]:


# Model Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
y_pred = regression.predict(x_test)
# MAE
mean_absolute_error(y_test,y_pred)


# In[32]:


# MSE
mean_squared_error(y_test,y_pred)


# In[33]:


#RMSE
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




