#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from azureml.core import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta

from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment

from sklearn.model_selection import train_test_split

import logging


# In[2]:


ws = Workspace.from_config()


# In[3]:


df = pd.read_csv('290321.csv')


# In[4]:


df = df.drop(columns=['status','Unnamed: 0'])


# In[5]:


df.info()


# In[6]:


x_train, x_test = train_test_split(df, test_size=0.2, random_state=223)
print(x_train['is_canc'].value_counts()/len(x_train))
print(x_test['is_canc'].value_counts()/len(x_test))


# In[7]:


automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours": 0.3,
    "enable_early_stopping": True,
    "primary_metric": 'norm_macro_recall',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 15
}


# In[8]:


automl_config = AutoMLConfig(task='classification',
                             debug_log='automated_ml_errors.log',
                             training_data=x_train,
                             label_column_name="is_canc",
                             **automl_settings)


# In[ ]:


experiment = Experiment(ws, "canc-090421")
local_run = experiment.submit(automl_config, show_output=True)


# In[ ]:


from azureml.widgets import RunDetails
RunDetails(local_run).show()


# In[ ]:


best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)


# In[ ]:


test = x_test.drop(columns=['is_canc'])
best_run, fitted_model = local_run.get_output()
class_prob = fitted_model.predict_proba(test)


# In[ ]:


class_prob


# In[ ]:


#pd.merge(class_prob, test, left_index=True, right_index=True)
pd.concat([class_prob, test], axis=1)


# In[ ]:


canc_prob = test.join(class_prob).sort_values(by=True, ascending=False).dropna(subset=[True])


# In[ ]:


canc_prob[True].value_counts(bins=2)


# In[ ]:


canc_prob[['property_id', 'room_type_name','tenancy_start_date','tenancy_end_date','total_price',True]]


# In[ ]:


canc_prob.info()


# In[ ]:




