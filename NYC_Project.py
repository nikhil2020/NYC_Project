#!/usr/bin/env python
# coding: utf-8

# # Project: 311 NYC service request. 

# # Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Importing the NYC Dataset 

# In[2]:


data = pd.read_csv('NYC.csv')

# Convert the Date columns into the datetime datatypes
# Below the Date columns is in object datatype intial
# In[3]:


data.dtypes


# In[ ]:


# Converting the Date columns into the datetime datatypes
data['Closed Date'] = pd.to_datetime(data['Closed Date'])
data['Created Date'] = pd.to_datetime(data['Created Date'])


# In[ ]:


data.dtypes


# In[ ]:



import datetime as dt


# In[ ]:


# Renaming the Created Date and Closed Date to Created_Date and Closed_Date
data.rename(columns={'Created Date': 'Created_Date','Closed Date':'Closed_Date','Complaint Type':'Complaint_Type','Location Type':'Location_Type'},inplace =True)


# In[ ]:


# The time elapsed betweeen the Created_Date and Closed_Date in seconds
time_elapsed = (data.Created_Date - data.Closed_Date).dt.total_seconds()


# In[ ]:


# The time in absolute as time can't be negative
t = abs(time_elapsed)


# In[ ]:


# Creating the new column Request_closed time in the dataset
data['Request_Closing_Time'] = t


# In[ ]:


data.head()


# # Data Understanding and Exploration
Here we are exploration and understanding the dataset
and finding the meaningful information from the dataset
# #### indo() is gives the information of structure of the dataset
# #### like how many rows and columns is in the dataset which variable
# #### is having the null value and what is the datatype of each variables
# 

# In[ ]:


data.info()


# #### Describe() gives the information of the statistic of each numerical variable
# #### like no. of count,mean of the column, standard deviation and min value
# #### 1st Quartile,2nd Quartile and 3rd Quartile and max value to see the insight of the dataset
# 

# In[ ]:


data.describe()


# In[ ]:


# head gives the top five rows how the dataset looks like
data.head()


# In[ ]:


# columns gives the name of each variables in the dataset
data.columns


# In[ ]:


# nunique() counts the no. of unique values in the columns
data.nunique()


# In[ ]:


# sns.swarmplot(x="Agency Name", y="Request_Closing_Time", data=data)


# #### Correlation heatmap show  the relation between the numerical variables
# #### Here in this map states thatn the location are highly corelated with each other
# 

# In[ ]:


sns.heatmap(data.corr())


# In[ ]:


# How many null values contains in each variables
data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


# pivot table to show the relation between Complaint Type and Agengy type with Request Closing Time
pv = data.pivot_table(values='Request_Closing_Time',index='Complaint_Type',columns='Agency Name')


# ## Analysis of  pivot table heatmap 
# #### heatmap plot show that Animal in a Park has the highest complaint through New york city police Department at request closing time above 1000000
# #### MOst of the complaint is coming through New York city police with an request closing time of 250000 seconds
# #### Blocked Driveway complaint is comes from NYPD with request closing time of 250000 seconds
# #### Agency Issues complaint is comes from NYPD with request closing time of 250000 seconds
# 

# In[ ]:


sns.heatmap(pv,cmap = 'summer')


# # Order the complaint types based on the average ‘Request_Closing_Time’ grouping them for different locations.
# 

# In[ ]:


d = data[['Complaint_Type','Request_Closing_Time','Location','Location_Type','Incident Address']]
d.head()


# In[ ]:


location_based=d.groupby(['Location_Type','Incident Address','Complaint_Type','Location'],as_index = True).mean()


# ### Different Location Based groupby with complaint type on the bases of aVerage Request closing time

# In[ ]:



location_based.head(30)


# # Hypothesis Testing
Perform statistical test for the following:
Please note: For the below statements you need to state the Null and Alternate and then provide a statistical test to accept or reject the Null Hypothesis along with the corresponding ‘p value’.
Whether the average response time across complaint types are similar or not (overall)
Is the type of complaint or service requested and location related?

# ## Hypothesis for First Statement:-
# Null Hypothesis : The average response time across complaint types are simliar
# #Alternate Hypothesis : The average response time across complaint types are not same

# In[ ]:


data.describe()


# In[ ]:


# Importing the stats model ols
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


#variable ~ treatment
mod = ols('Request_Closing_Time~Complaint_Type', data=data).fit()


# In[ ]:


data.Request_Closing_Time.isnull().sum()


# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
data[['Request_Closing_Time' ]]=imputer.fit_transform(data[['Request_Closing_Time' ]])


# In[ ]:


data.Request_Closing_Time.isnull().sum()


# In[ ]:


# Summary of the anova test
mod.summary()


# In[ ]:


aov_table = sm.stats.anova_lm(mod)


# In[ ]:


aov_table


# In[ ]:


# double click for zoom the plot show the variance relation between  Request closing time and complaint type
data.boxplot('Request_Closing_Time',by = 'Complaint_Type',figsize = (50,10),rot =0)


# In[ ]:


# Tukey method show 


# In[ ]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[ ]:


tukey = pairwise_tukeyhsd(data.Request_Closing_Time, data.Complaint_Type, alpha = 0.05)


# In[ ]:


tukey.summary()


# In[ ]:


# Plot show the variance of the complaint type is similar with respective to Request closing time
tukey.plot_simultaneous(figsize = (15,8))
plt.show()

By tukey method show the null hypothesis is accepted as there is similar average request closing time to the Complaint type

# # #Is the type of complaint or service requested and location related?
# Null Hypothesis : The type of Complaint or service requested are 
#     related to the location.
# 
# Alternate Hypothesis : The type of Complaint or service requested are
#     not related to the location.

# In[ ]:


contigency_table = pd.crosstab(data.Complaint_Type,data.Location_Type)
contigency_table


# In[ ]:


import pandas as pd
import scipy.stats as stats
from math import sqrt


# In[ ]:


print(stats.chi2_contingency(contigency_table))


# In[ ]:


chi_square , p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(contigency_table)


# In[ ]:


chi_square, p_value


# In[ ]:


chi_square


# In[ ]:


p_value


# In[ ]:


degrees_of_freedom


# In[ ]:


expected_frequencies

Here from the above observation we state that null hypothesis is accepted as there is relation between 
complaint type and location