#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[30]:


#Import the dataset 
dataset = pd.read_csv('term-deposit-marketing-2020.csv')


# In[31]:


dataset


# In[23]:


dataset.dtypes


# In[24]:


#mapping nonnumeric values to numbers
nu_col=dataset._get_numeric_data().columns
liste=list(set(col)-set(nu_col))
for i in liste:
    dataset[i] = dataset[i].rank(method='dense', ascending=True).astype(int)
print(dataset)


# In[8]:


#splitting train data and target data
X= dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values


# In[9]:


#Train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[10]:


#Training a decision tree model
tree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)


# In[11]:


#Get the predictions
Y_pred = tree.predict(X_test)


# In[12]:


#Average performance scores
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

results = confusion_matrix(y_test, Y_pred)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(y_test, Y_pred))
print ('Classification Report : ')
print (classification_report(y_test, Y_pred))
print('AUC-ROC:',roc_auc_score(y_test, Y_pred))


# In[14]:


#average accuracy over 5-fold cross validation
from sklearn.model_selection import KFold
scores = []
cv = KFold(n_splits=5, random_state=0, shuffle=True)
for train_index, test_index in cv.split(X):
   # print("Train Index: ", train_index, "\n")
   # print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    tree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)
    scores.append(tree.score(X_test,y_test))


# In[15]:


average_accuracy=np.mean(scores)
average_accuracy


# In[29]:


#duration(last contact duration) is most correlated feature with y(has the client subscribed to a term deposit)
#at a rate -0.46. We should focun on duration feature. I doesn't matter to be a negative or positive value
#absolute value represent how much they are related to each other.
maxi=0
for i in set(dataset.columns):
    cor=dataset['y'].corr(dataset[i])
    if np.abs(cor)>maxi and cor!=1:
        maxi=np.abs(cor)
        m=i
print(m ,":", dataset['y'].corr(dataset[m]))
print("max",maxi)

