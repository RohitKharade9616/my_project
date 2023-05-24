#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[43]:


# importing dataset
data=pd.read_csv("crop.csv")


# In[44]:


print(data.head())


# In[45]:


print(data.describe())


# In[46]:


print(data.isna().sum())


# In[47]:


data.nunique()


# In[48]:


data['CROP'].unique()


# In[49]:


print(data.columns)


# In[50]:


crop_summary=pd.pivot_table(data,index=["CROP"],aggfunc='mean')


# In[51]:


print(crop_summary)


# In[58]:


x=data.iloc[:,:7]
# print(x.head())
y=data.iloc[:,7]
#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  


# In[69]:



correlation = data. corr ()  
cr=sns. heatmap (correlation) 
cr.show()


# In[62]:


y_pred= classifier.predict(x_test) 


# In[63]:


print(y_pred)


# In[64]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred , y_test)
print(accuracy)


# In[65]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(15,15))
sns.heatmap(cm,annot=True,fmt=".0f",linewidths=.5,square=True,cmap="Blues")
plt.ylabel("actual label")
plt.xlabel("predicted label")
all_sample_title="Confusion Matrix - score :"+str(accuracy_score(y_test,y_pred))
plt.title(all_sample_title,size=15)
plt.show()


# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




