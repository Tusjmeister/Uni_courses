#!/usr/bin/env python
# coding: utf-8

# # DAT200 CA3 2022
# 
# Kaggle username: Arterx

# ### Imports

# In[57]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# ### Reading data

# In[23]:


raw_data = pd.read_csv('train.csv') # Naming the train data "raw_data"
test_data = pd.read_csv('test.csv') # Naming the test data "test_data"


# ### Data exploration and visualisation

# In[3]:


# Histograms below

raw_data.hist()
plt.tight_layout()
plt.show()

# Pairplots below
sns.pairplot(raw_data, hue='target')


# In[4]:


raw_data.describe() # Descriptive stats


# ### Data cleaning

# In[50]:


# Using Z-scores to filter out the outliers. Z-score < 2 is 95% of the data

z_scores = stats.zscore(raw_data)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 2).all(axis=1)

# outliers = (abs_z_scores >= 2).all(axis=1) Was used with the IterativeImputer but did not prove useful

data = raw_data[filtered_entries]


# In[51]:


data.describe() # Descriptive stats


# ### Data exploration after cleaning

# In[49]:


# Histograms below

data.hist()
plt.tight_layout()
plt.show()


sns.pairplot(data, hue='target')

# We see that the z-score method removed the worst outliers.


# ### Data preprocessing

# In[84]:


X = data.iloc[:, :-1]
y = data.target


# #### Train test split

# In[85]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# #### Scaling

# In[83]:


sc = StandardScaler()
sc.fit(X_train)


# Transform (standardise) both X_train and X_test with mean and STD from
# training data
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

# I did not use the scaled data at first, but after some testing, I ended up with a slightly better result when using the scaled data.
# Furthermore, I tried to do a grid search but it took me 4 hours and did not increase the accuracy by much
# Below I am using the relevant values I had gotten from the grid search


# ### Modelling

# In[ ]:


forest = RandomForestClassifier(criterion='entropy',
                                max_features='auto',
                                n_estimators=250,
                                random_state=1,
                                n_jobs=-1,
                                bootstrap=True
                               )

forest.fit(X_train_sc, y_train)


# ### Evaluation

# In[87]:


print('Forest training data accuracy: {0:.2f}'.format(forest.score(X_train_sc, y_train)))

print('Forest test data accuracy: {0:.2f}'.format(forest.score(X_test_sc, y_test)))


# ### Kaggle submission

# In[81]:


pred_forest = forest.predict(X_test_sc)

score_forest = f1_score(y_test, pred_forest, average='weighted')
print(f'\nRandom Forest Score: {score_forest}\n')


# In[82]:


forest = RandomForestClassifier(criterion='entropy',
                                max_features='auto',
                                n_estimators=250,
                                random_state=1,
                                n_jobs=-1,
                               bootstrap=True)
forest.fit(X, y)

# Predicting and inserting the results into a file named "submission_n"

submission_16 = forest.predict(test_data)
df_submission_16 = pd.DataFrame(submission_16)
df_submission_16.reset_index(level=0, inplace=True)

df_submission_16.columns = ['index', 'target']
df_submission_16.to_csv('submission_16.csv', index=False)

