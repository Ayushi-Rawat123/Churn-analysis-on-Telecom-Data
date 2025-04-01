#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install numpy


# In[3]:


pip install matplotlib


# In[4]:


pip install seaborn


# In[5]:


import pandas as pd


# In[6]:


import numpy as np


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


import seaborn as sns


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv("C:\\Users\\Lenovo\\Searches\\Downloads\\Churn analysis Python\\Customer-Churn-analysis-main\\Customer Churn.csv")
df.shape
df


# In[11]:


df.info()


# #### Replacing the blank columns with 0 as tenure is 0 no total charges are recorded.
# #### Changing the Data type of total charges from object to float.

# In[12]:


df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype("float")


# In[13]:


df.info()


# In[14]:


df.isnull().sum().sum()


# In[15]:


df.describe()


# In[16]:


df.duplicated().sum()


# In[17]:


df.duplicated('customerID').sum()


# #### Convert 0 and 1 vaules of senior citizer column to boolean values to make it easy to understand.

# In[18]:


def conv(value):
    if value == 1:
        return "Yes"
    else:
        return "No"
    
df['SeniorCitizen'] = df['SeniorCitizen'].apply(conv)


# In[19]:


df.head()


# In[20]:


df.tail(10)


# In[21]:


ax = sns.countplot(x='Churn', data = df)
ax.bar_label(ax.containers[0])
colors = ['#2c3e50', '#8e44ad']
plt.title('Count of Cistomers by Churn')
plt.figure(figsize = (4, 4))
plt.show()


# In[22]:


plt.figure(figsize = (3, 4))
gb = df.groupby('Churn').agg({'Churn': "count"})
colors = ['#1abc9c', '#3498db']
plt.pie(gb['Churn'], labels = gb.index, autopct = '%1.2f%%', colors = colors)
plt.title('Percentage of Customers by Churn')
plt.show()


# #### From the given Pie chart above we can analyze that 26.54% out of 100% of our customers have churned out.

# In[23]:


plt.figure(figsize = (4, 4))
sns.countplot(x = 'gender', data = df, hue= 'Churn')
plt.title('Churn by Gender')
plt.show()


# #### From the chart above we have analysed that Churn is not gender based , almost equal number of male and female churned out.

# In[24]:


plt.figure(figsize = (4, 4))
sns.countplot(x = 'SeniorCitizen', data = df, hue= 'Churn')
plt.title('Churn by Senior Citizens')
plt.show()


# In[25]:


plt.figure(figsize = (4, 5))
ax = sns.countplot(x = 'SeniorCitizen', data = df)
ax.bar_label(ax.containers[0])
plt.title('Churn Count by Senior Citizens')
plt.show()


# In[26]:


counts = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100

plt.figure(figsize=(5, 5))
counts.plot(kind='bar', stacked=True, colormap='Wistia', ax=plt.gca())

for index, (category, row) in enumerate(counts.iterrows()):
    cumulative = 0  
    for churn_category, value in row.items():
        plt.text(index, cumulative + value / 2, f"{value:.1f}%", ha='center', fontsize=10, color='black')
        cumulative += value 

plt.title('Churn by Senior Citizens (%)')
plt.xlabel('Senior Citizen')
plt.ylabel('Percentage')
plt.xticks(rotation = 0)
plt.legend(title='Churn', bbox_to_anchor = (0.9, 0.9))
plt.show()


# #### Comparetively a greater percentage of people in senior citizen category have churned out.

# In[27]:


plt.figure(figsize = (10, 6))
sns.histplot(x = 'tenure', data = df, bins = 72, hue = 'Churn')
plt.show()


# #### People who have used our services for a long time have stayed and people who have used our services only for 1 or 2 months have churned.

# In[28]:


colors = ['#ff6b6b', '#feca57'] 
plt.figure(figsize=(4, 5))
ax = sns.countplot(x='Contract', data=df, palette=colors, hue = 'Churn')  
ax.bar_label(ax.containers[0])
plt.title('Churn by Contract', color='#333') 
plt.show()


# #### People who have month - month contract are likely to churn than those who have 1 or 2 years of contract.

# In[29]:


plt.figure(figsize=(6, 5))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, palette=colors)
plt.title("Distribution of Monthly Charges by Churn", color='#333')
plt.show()


# In[30]:


cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 14)) 
axes = axes.flatten()
for i, col in enumerate(cols):
    sns.countplot(x=df[col], ax=axes[i], palette='coolwarm', hue = df["Churn"])  
    axes[i].set_title(col, fontsize=12, fontweight='bold')  
    axes[i].tick_params(axis='x', rotation=0)  

plt.tight_layout()  
plt.show()


# #### The subplots analyze customer churn based on various telecom services. Higher churn is observed among customers lacking online security, backup, tech support, and those using fiber optic internet. Streaming services (TV & Movies) show no significant impact on churn. Improving security and support services could help reduce customer attrition.

# In[31]:


plt.figure(figsize = (6, 4))
ax = sns.countplot(x = 'PaymentMethod', data = df, hue = 'Churn')
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.xticks(rotation = 45)
plt.title('Churned Count by Payment Method')
plt.show()


# #### Customers to likely churn when ther are using Electronic payment method.

# In[32]:


cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))  
axes = axes.flatten()  

for i, col in enumerate(cols):
    if i % 3 == 0:
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values, ax=axes[i], palette='coolwarm')
        axes[i].set_ylabel("Count")
        
    elif i % 3 == 1:
        df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=axes[i], colors=sns.color_palette("coolwarm", 3))
        axes[i].set_ylabel("")  

    else:
        try:
            sns.kdeplot(data=df, x=col, hue="Churn", fill=True, palette="coolwarm", ax=axes[i])
            axes[i].set_ylabel("Density")
        except:
            sns.countplot(x=df[col], hue=df["Churn"], palette="coolwarm", ax=axes[i])

    axes[i].set_title(col, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# #### The above subplot is also comparision of various other factors using different bar charts.

# In[ ]:





# In[ ]:





# # Overall Summary
# 
# ## This project focuses on churn analysis in the telecom industry, utilizing customer data to identify key factors influencing customer retention. The dataset is processed by handling missing values, converting data types, and ensuring accuracy.
# 
# ### Exploratory Data Analysis (EDA) reveals that:
# 
# Churn Rate: Around 26.54% of customers have churned.
# 
# Contract Type: Month-to-month contract users are more likely to churn.
# 
# Senior Citizens: Higher churn rate compared to younger customers.
# 
# Tenure: New customers (1-2 months) have a high churn rate, whereas long-term users tend to stay.
# 
# Monthly Charges: Higher charges correlate with increased churn.
# 
# Payment Method: Customers using electronic checks show higher churn tendencies.
# 
# Online Services: Lack of online security, backup, and tech support leads to more churn.

# In[ ]:




