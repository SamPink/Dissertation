#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,recall_score
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt  

import xgboost as xgb

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 400)


# In[2]:


#importing bookings dataset
bo = pd.read_csv('290321.csv')


# In[3]:


drop = ['tenancy_start_date', 'tenancy_end_date', 'status', 'status_time_applied'
        , 'status_time_room_selected', 'status_time_selection_completed', 'status_time_details_completed'
        , 'status_time_terms_accepted', 'created_at']

bo = bo.drop(columns=drop)


# In[4]:


num_vars = ['property_id','tenancy_length','price_per_night','age'
            ,'hours_to_room_selected','hours_to_terms_accepted','hours_to_start_date'
            ,'hours_to_end_date','hours_to_selection_completed','hours_to_details_completed']

cat_vars = ['source','room_type_name','device','installment_type','is_rebooker','gender'
            ,'nationality','destination_university','year_of_study','major'
            ,'communication_preferences','heard_source','degree_classification'
            ,'academic_year']


# In[5]:


df = pd.get_dummies(bo,columns=cat_vars)


# In[6]:


imr = SimpleImputer(strategy='mean')
df[num_vars] = imr.fit_transform(df[num_vars])


# In[7]:


scaler = StandardScaler()

df[num_vars] = scaler.fit_transform(df[num_vars])
df = pd.DataFrame(df)


# In[8]:


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]


# In[9]:


X= df.drop(columns='is_canc')
y = df['is_canc']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


# In[11]:


xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)


# In[12]:


y_pred = xgb_clf.predict(X_test)


# In[13]:


print(accuracy_score(y_test, y_pred))


# In[14]:


ft_imp = pd.Series(xgb_clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
ft_imp[0:15].plot.barh()


# In[26]:


ft_imp[0:11]


# In[33]:


confusion_matrix(y_test, y_pred)


# In[32]:


C.astype('float') / C.sum(axis=1)[:, np.newaxis]


# In[16]:


metrics.precision_score(y_test, y_pred)


# In[35]:


recall_score(y_test, y_pred)


# In[17]:


f1_score(y_test, y_pred, average='weighted')


# In[18]:


metrics.plot_roc_curve(y_test, y_pred)  
plt.show()     

