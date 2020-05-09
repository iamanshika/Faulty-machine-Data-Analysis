#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# ## Data Exploration

# In[2]:


#importing the csv file in dataframe
df = pd.read_csv("maintenance_data.csv")
df


# In[3]:


df.head()


# In[4]:


#Understanding data domain
df.info()


# The temperature and pressure columns have 3 and 4 missing values respectively.

# In[5]:


#To check if there are any outliers in the data
df.describe()


# There are signs that there might be outliers in lifetime column and moisture column.

# ## Data Cleaning

# In[6]:


#To check if any other character other than NaN is used to depict Null values
print(df['broken'].unique())
print(df['team'].unique())
print(df['provider'].unique())


#  This shows that no special characters are used to depict NaN values.

# In[7]:


#To check the total number of missing values
df.isnull().sum()


# The 'pressureInd' column has 4 Null values and 'temperatureInd' column has 3 Null values

# In[8]:


#To check if there are any duplicted entries
df.duplicated().sum()


# There are no duplicated entries.

# In[9]:


#To decide if the NaN values sholud be replaced with the mean or the median.
df.skew()


# We can fill the missing values in pressure and temperature columns by their mean as the data in not much skewed

# In[10]:


#Replacing NaN values with the mean 
df.pressureInd.fillna(df.pressureInd.mean(), inplace = True )
df.temperatureInd.fillna(df.temperatureInd.mean(),inplace = True)
df.info()


# In[11]:


#removing the outliers
df.drop(df[df['moistureInd']>200].index,inplace = True)
df.describe()


# ## Data Analytics

# #### Univariate Analysis

# In[12]:


#Categorising the columns into numeric or categorical
numerics = ['lifetime', 'moistureInd', 'pressureInd', 'temperatureInd']
categorical = ['broken', 'team', 'provider']


# In[13]:


#Using Matplotlib's Scatter plot to depict univariate analysis of numeric data
for i in numerics:
    plt.figure(figsize = (12,5))
    plt.scatter(np.arange(999),df[i])
    plt.title(i)
    plt.show()


# All graphs show uniform distribution except the mositure graph. Most of the machines are wroking at lower moisture level and a very few are working at high moisture levels. This might be a problem.

# In[14]:


#Using Seaborn's Countplot to depict univariate analysis of categorical data
for i in categorical:
    plt.figure(figsize = (12,5))
    sb.countplot(df[i])
    plt.title(i)
    plt.show()


# #### Bivariate Analysis

# In[15]:


#Using Matplotlib's Distribution Plot to depict Numeric v/s Categorical data
#Moisture v/s Broken
plt.figure(figsize = (12,5))
sb.distplot(df.moistureInd[df.broken==0])
sb.distplot(df.moistureInd[df.broken==1])
plt.legend(['0','1'])
plt.show()


# The distribution depicts that the machines wroking at moisture levels higher than 125 are more likely to be broken.

# In[16]:


#Temperature v/s Broken 
plt.figure(figsize = (12,5))
sb.distplot(df.temperatureInd[df.broken==0])
sb.distplot(df.temperatureInd[df.broken==1])
plt.legend(['0','1'])
plt.show()


# The distribution tends to depict that temperature as such does not affect the number of broken and not broken machines as both the broken and not broken machines operate at same temperature levels.

# In[17]:


#Pressure v/s Broken
plt.figure(figsize = (12,5))
sb.distplot(df.pressureInd[df.broken==0])
sb.distplot(df.pressureInd[df.broken==1])
plt.legend(['0','1'])
plt.show()


# The distribution tends to depict that pressure as such does not affect the number of broken and not broken machines as both the broken and not broken machines operate at same pressure levels.

# In[18]:


#Lifetime v/s Broken
plt.figure(figsize = (12,5))
sb.distplot(df.lifetime[df.broken==0])
sb.distplot(df.lifetime[df.broken==1])
plt.legend(['0','1'])
plt.show()


# The distribution plot is depicting that the machines which have a lifetime for 60 months or more are likely to be broken.

# In[19]:


#Using Matplotlib's Countplot to  depict Categorical v/s Categorical data
#Team v/s Broken
plt.figure(figsize = (12,5))
sb.countplot(df.team)
plt.show()
plt.figure(figsize = (12,5))
sb.countplot(df.team[df.broken==1])
plt.show()


# Team C maintains their machines slightly less.

# In[20]:


#Provider v/s Broken
plt.figure(figsize = (12,5))
sb.countplot(df.provider)
plt.show()
plt.figure(figsize = (12,5))
sb.countplot(df.provider[df.broken==1])
plt.show()


# The countplot depicts that Provider 4 is providing the best machines and Provider 1 and Provider 3, the worst.

# ## Multivariate Analysis

# In[21]:


#Using Seaborn's point plot to depict Numeric v/s Categorical v/s Categorical Data
plt.figure(figsize = (12,5))
sb.pointplot(x = 'provider', y = 'lifetime', hue = 'broken', data = df)
plt.show()


# The point plot is depicting that Provider 2 is providing the best machines and Provider 3 the worst.

# In[22]:


plt.figure(figsize = (12,5))
sb.pointplot(x = 'team', y = 'lifetime', hue = 'broken', data = df)
plt.show()


# The point plot is depicting that the Team C manages the machines slightly less.

# In[23]:


#Plotting Seaborn's Swarm plot to draw better conclusions for the above plotted graph
plt.figure(figsize = (12,5))
sb.swarmplot(x = 'team',y = 'lifetime', hue = 'broken' , data = df)
plt.show()


# The point plot is depicting that the Team C has slightly less durable machines.

# In[24]:


#Using Seaborn's heatmap to depict the linear relationships between the features
#Plotting the heatmap
corr = df.corr() #Calculating the correlation matrix
plt.figure(figsize = (12,8))
sb.heatmap( corr, annot = True, cmap = 'coolwarm')
plt.show()


# The heatmap is clearly depicting linear relationships between [Moisture, Lifetime] and [Moisture, Broken]. There is a non-linear relationship existing between [Moisture, Lifetime] which has already been shown

# ### Ans1. The machines which have a lifetime for 60 months or more are likely to be broken.
# ### Ans2. Provider 2 is providing the best machines and Provider 3 the worst. 
# ###       Team C manages the machines slightly less.  
# ###        Some machine providers are better than others and some teams are better at machine management than the rest.

# In[ ]:




