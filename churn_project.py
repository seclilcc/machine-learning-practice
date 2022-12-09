#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
 
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[4]:


len(df)


# In[5]:


df.head()


# In[6]:


df.head().T


# In[7]:


df.dtypes


# In[8]:


total_charges = pd.to_numeric(df.TotalCharges, errors='coerce')


# In[9]:


df[total_charges.isnull()][['customerID', 'TotalCharges']]


# In[10]:


df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
df.TotalCharges = df.TotalCharges.fillna(0)


# In[11]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
 
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
 
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[12]:


df.churn = (df.churn == 'yes').astype(int)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[15]:


df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)
 
y_train = df_train.churn.values
y_val = df_val.churn.values
 
del df_train['churn']
del df_val['churn']


# In[16]:


df_train_full.isnull().sum()


# In[17]:


df_train_full.churn.value_counts()


# In[18]:


global_mean = df_train_full.churn.mean()


# In[19]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[20]:


df_train_full[categorical].nunique()


# In[21]:


female_mean = df_train_full[df_train_full.gender == 'female'].churn.mean()


# In[22]:


male_mean = df_train_full[df_train_full.gender == 'male'].churn.mean()


# In[23]:


partner_yes = df_train_full[df_train_full.partner == 'yes'].churn.mean()
partner_no = df_train_full[df_train_full.partner == 'no'].churn.mean()


# In[24]:


print('gender == female:', round(female_mean, 3))
print('gender == male:', round(male_mean, 3))


# In[25]:


print('partner == yes', round(partner_yes, 3))
print('partner == no', round(partner_no, 3))


# In[26]:


global_mean = df_train_full.churn.mean()
 
df_group = df_train_full.groupby(by='gender').churn.agg(['mean'])
df_group['diff'] = df_group['mean'] - global_mean
df_group['risk'] = df_group['mean'] / global_mean
 
df_group


# In[27]:


from IPython.display import display 
 
for col in categorical:
    df_group = df_train_full.groupby(by=col).churn.agg(['mean'])
    df_group['diff'] = df_group['mean'] - global_mean
    df_group['rate'] = df_group['mean'] / global_mean
    display(df_group)


# In[28]:


from sklearn.metrics import mutual_info_score
 
def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)
 
df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
df_mi


# In[29]:


df_train_full[numerical].corrwith(df_train_full.churn)


# In[30]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')


# In[31]:


from sklearn.feature_extraction import DictVectorizer
 
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)


# In[32]:


X_train = dv.transform(train_dict)


# In[33]:


X_train[0]


# In[34]:


dv.get_feature_names()


# In[35]:


def logistic_regression(xi):
    score = bias
    for j in range(n):
        score = score + xi[j] * w[j]
    prob = sigmoid(score)
    return prob


# In[36]:


import math
 
def sigmoid(score):
    return 1 / (1 + math.exp(-score))


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)


# In[39]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[40]:


y_pred = model.predict_proba(X_val)


# In[41]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[42]:


y_pred >= 0.5


# In[43]:


churn = y_pred >= 0.5


# In[44]:


(y_val == churn).mean()


# In[45]:


dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))


# In[46]:


small_subset = ['contract', 'tenure', 'totalcharges']
train_dict_small = df_train[small_subset].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small)
 
X_small_train = dv_small.transform(train_dict_small)


# In[47]:


dv_small.get_feature_names()


# In[48]:


model_small = LogisticRegression(solver='liblinear', random_state=1)
model_small.fit(X_small_train, y_train)


# In[49]:


model_small.intercept_[0]


# In[50]:


dict(zip(dv_small.get_feature_names(), model_small.coef_[0].round(3)))


# In[51]:


customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75,
}


# In[52]:


X_test = dv.transform([customer])


# In[53]:


model.predict_proba(X_test)


# In[54]:


model.predict_proba(X_test)[0, 1]


# In[55]:


customer = {
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'phoneservice': 'yes',
    'multiplelines': 'yes',
    'internetservice': 'fiber_optic',
    'onlinesecurity': 'no',
    'onlinebackup': 'no',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'yes',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 85.7,
    'totalcharges': 85.7
}


# In[56]:


X_test = dv.transform([customer])
model.predict_proba(X_test)[0, 1]

