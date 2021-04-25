#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyodbc
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
import sys
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 400)
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sn
from datetime import date, timedelta
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})


# In[ ]:


username = 'GSA_Admin'
password = '92eCt2iPaFHjkBeX'
server = 'gsa-dev-sql-data-factory.database.windows.net'
database = 'raw-studentsuite'


# In[ ]:


URL = "mssql+pyodbc://" + username + ":" + password + "@" + server + "/" + database + "?driver=ODBC+Driver+17+for+SQL+Server"

sql_conn = create_engine(URL,fast_executemany=True).connect()


# In[ ]:


bo_cols = ['id','student_id','property_id','academic_year_id'
           ,'source','room_type_name','tenancy_start_date','tenancy_end_date'
           ,'tenancy_length','price_per_night','total_price','status','status_time_applied'
           ,'status_time_room_selected','status_time_selection_completed','status_time_details_completed'
           ,'status_time_terms_accepted','device','created_at','installment_type','is_rebooker']
bo = pd.read_sql("bookings.bookings", sql_conn, columns=bo_cols)


# In[ ]:


stu_cols = ['id','date_of_birth','gender','nationality','destination_university','year_of_study'
            ,'major','communication_preferences','heard_source','degree_classification']
stu = pd.read_sql("bookings.students", sql_conn, columns=stu_cols)


# In[ ]:


acaYear = pd.read_sql('configurations.academic_years', sql_conn, columns=['id','name']).rename(columns={"name": "academic_year"})


# In[ ]:


df = bo.join(
    stu.set_index('id'), on='student_id', lsuffix='_booking', rsuffix='_student'
).join(
    acaYear.set_index('id'), on='academic_year_id', lsuffix='_booking', rsuffix='_acaYear'
)


# In[ ]:


df = df.rename(columns={"name": "property_name"})


# In[ ]:


to_dt = ['tenancy_start_date','tenancy_end_date','status_time_applied','status_time_room_selected'
         ,'status_time_selection_completed','status_time_details_completed','status_time_terms_accepted'
         ,'date_of_birth']

df[to_dt] = df[to_dt].apply(pd.to_datetime, format='%Y-%m-%d',errors='coerce')


# In[ ]:


# add age from D.O.B 
today = date.today()
df['age'] = df['date_of_birth'].apply(lambda x: today.year - x.year - ((today.month, today.day) <(x.month, x.day)))


# In[ ]:


#calculate hours taken for status updates
t0 = df['status_time_applied']
    
df['hours_to_room_selected'] = (df['status_time_room_selected'] - t0).astype('timedelta64[h]')
df['hours_to_terms_accepted'] = (df['status_time_terms_accepted'] - t0).astype('timedelta64[h]')
df['hours_to_start_date'] = (df['tenancy_start_date'] - t0).astype('timedelta64[h]')
df['hours_to_end_date'] = (df['tenancy_end_date'] - t0).astype('timedelta64[h]')
df['hours_to_selection_completed'] = (df['status_time_selection_completed'] - t0).astype('timedelta64[h]')
df['hours_to_details_completed'] = (df['status_time_details_completed'] - t0).astype('timedelta64[h]')


# In[ ]:


#remove any booking with invalid price
df = df[df['total_price'].notna()]
df = df.drop(df[df['total_price'] < 1].index)


# In[ ]:


#remove any booking with invalid age
df = df[df['age'].notna()]
df = df.drop(df[df['age'] < 10].index)


# In[ ]:


#remove any booknig that did not reach terms accepted
df = df[df['status_time_terms_accepted'].notna()]


# In[ ]:


#remove any booking made before it started
df = df.drop(df[df['hours_to_selection_completed'] < 0].index)
df = df.drop(df[df['hours_to_terms_accepted'] < 0].index)
df = df.drop(df[df['hours_to_start_date'] < 0].index)


# In[ ]:


df.installment_type = df.installment_type.fillna('Other')
df.gender = df.gender.fillna('Other')
df.nationality = df.nationality.fillna('Other')
df.destination_university = df.destination_university.fillna('Other')
df.year_of_study = df.year_of_study.fillna('Other')
df.major = df.major.fillna('Other')
df.heard_source = df.heard_source.fillna('Other')
df.degree_classification = df.degree_classification.fillna('Other')


# In[ ]:


df['is_canc'] = df['status'].str.contains('cancelled')


# In[ ]:


df = df.drop(columns=['id','student_id','academic_year_id', 'date_of_birth'])


# In[ ]:


df.to_csv('290321.csv')

