#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics     # Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv")
pima.columns = col_names
pima.head()

#split dataset in features and target variable
#feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
X = pima[feature_cols]  # Features
y = pima.label  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)    # 70% training and 30% test

# default (criterion=”gini”)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred),"\n")

clf = RandomForestClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(random_state=2)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(max_depth=4)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred),"\n")

clf = RandomForestClassifier(criterion="entropy", max_depth=4, random_state=2)  # max_depth = 4, random_state = 2
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print("train Accuracy:",metrics.accuracy_score(y_train, y_pred))
y_pred = clf.predict(X_test)
print("test Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[2]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0,tol=0.0001).fit(X_train, y_train)   # random_state = 0, tol = 0.001
print(clf.score(X_train, y_train), clf.score(X_test, y_test))

for i in feature_cols:
    tmp = X_train.drop(i,1)
    tmp1 = X_test.drop(i,1)
    clf = LogisticRegression(random_state=0,tol=0.001).fit(tmp, y_train)
    print(clf.score(tmp, y_train), clf.score(tmp1, y_test))

