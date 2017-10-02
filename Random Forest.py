
# coding: utf-8

# In[53]:


# Imports
import pandas as pd # load csv's (pd.read_csv)
import numpy as np # math (lin. algebra)

import random

from sklearn.ensemble import RandomForestClassifier

# Visualisation
import matplotlib.pyplot as plt # plot the data
import seaborn as sns # data visualisation
sns.set(color_codes=True)
#% matplotlib inline

import util


# In[54]:


x_raw,y_raw=util.loadTrainData()
x_test=util.loadTestData()


# In[55]:


table={"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
y_temp=[]
for label in y_raw:
    y_temp.append(table[label])
x=x_raw
y=np.array(y_temp)


# In[56]:


num_samples = x.shape[0] # number of features
print("Number of all samples: \t\t", num_samples)
split = int(num_samples * 2/3)


# In[57]:


x_validation=x[split:]
x_train=x[:split]
y_validation=y[split:]
y_train=y[:split]


# In[58]:


clf=RandomForestClassifier(n_estimators=300, max_features='auto', max_depth=40, n_jobs=-1)
model=clf.fit(x_train,y_train)


# In[59]:


y_predict=model.predict_proba(x_validation)


# In[61]:


logloss=util.evaluation(y_predict,y_validation)


# In[62]:


print(logloss)


# In[42]:


y_test_predict=model.predict_proba(x_test)


# In[13]:


util.writeOut(y_test_predict)


# In[ ]:




