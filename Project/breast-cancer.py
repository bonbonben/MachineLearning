#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# preprossing is what we do with the data before we run the learning algorithm
from sklearn import preprocessing
# taking included data set from Sklearn http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import linear_model


# In[2]:


# logistic regression sklearn
cancer = load_breast_cancer()
y = cancer.target
X = cancer.data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Append a column of ones to x_train
# Step 1: Create a column vector of ones (i.e. a vector of shape N',1)
ones = np.ones(X_train.shape[0]).reshape((X_train.shape[0], 1))
# Step 2: Append a column of ones in the beginning of x_train
X_train = np.hstack((ones, X_train))

# Now do the same for the test data
# Step 1: Create a column vector of ones (i.e. a vector of shape N",1)
ones = np.ones(X_test.shape[0]).reshape((X_test.shape[0], 1))
# Stemp 2: Append a column of ones in the beginning of x_test
X_test = np.hstack((ones, X_test))


# In[3]:


# Logistic Regression using sklearn feature selection
clf = LogisticRegression(random_state=0,tol=0.001).fit(X_train, y_train)   # random_state = 0, tol = 0.001
base_train = clf.score(X_train, y_train)
base_test = clf.score(X_test, y_test)
print("Logistic Regression:")
print("train Accuracy:",base_train)
print("test Accuracy:",base_test,"\n")

# feature extraction
for i in range(31):
    tmp = np.delete(X_train, i, axis = 1)
    tmp1 = np.delete(X_test, i, axis = 1)
    clf = LogisticRegression(random_state=0,tol=0.001).fit(tmp, y_train)
    fs_train = clf.score(tmp, y_train)
    fs_test = clf.score(tmp1, y_test)
    if fs_train > base_train:
        if fs_test >= base_test:
            print("Feature Selection:")
            print("remove",i)
            print("train Accuracy:",fs_train)
            print("test Accuracy:",fs_test,"\n")


# In[4]:


# Logistic Regression from scratch
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

w = np.zeros((X_train.shape[1], 1))
print(w.shape)

def hypothesis(X, w):
    #TODO
    h = np.ones((X.shape[0],1))
    for i in range(0,X.shape[0]):
        z = np.matmul(X[i], w)
        h[i] = sigmoid(z)
    return h

# Compute y_hat using our training examples and w (w is still set to zero). 
# This is just a preliminary test of the hypotheis function
yhat = hypothesis(X_train, w)

# print the sizes of yhat and y as a first check that the function performed correctly
print(yhat.shape) # this should return (426, 1)
print(y_train.shape) # this should return (426,)

def log_likelihood(X, y, w):
    ##TODO
    log_likelihood = 0
    for i in range(0,X.shape[0]):
        z = np.matmul(X[i], w)
        log_likelihood += y[i] * np.log(sigmoid(z)) + (1 - y[i]) * np.log((1 - sigmoid(z)))
    ##
    return log_likelihood

def Logistic_Regresion_Gradient_Ascent(X, y, learning_rate, num_iters):
    # For every 100 iterations, store the log_likelihood for the current w
    # Initializing log_likelihood to be an empty list
    log_likelihood_values = []
    # Initialize w to be a zero vector of shape x_train.shape[1],1
    w = np.zeros((X.shape[1], 1))
    # Initialize N to the number of training examples
    N = X.shape[0]
    ## TODO 
    for i in range (num_iters):
        y_hat = hypothesis(X, w)
        temp = 0
        for j in range (0,N):
            temp += (y[j] - y_hat[j]) * X[j]
        for k in range (0,w.shape[0]):
            w[k] = w[k] + (learning_rate / N) * temp[k]
    ##
        if (i % 100) == 0:
            log_likelihood_values.append(log_likelihood(X,y,w))
        
    return w, log_likelihood_values
learning_rate = 0.1
num_iters = 100 # The number of iteratins to run the gradient ascent algorithm
w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_train, y_train, learning_rate, num_iters)


# In[5]:


def predict(X_test, w):
    pred = hypothesis(X_test, w)
    res = []
    for i in range(0,len(pred)):
        if pred[i] >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

train_pred = predict(X_train, w)
test_pred = predict(X_test, w)
print("Logistic Regression:")
print("train Accuracy:",metrics.accuracy_score(y_train, train_pred))
print("test Accuracy:",metrics.accuracy_score(y_test, test_pred),"\n")

# feature extraction
for i in range(31):
    tmp = np.delete(X_train, i, axis = 1)
    tmp1 = np.delete(X_test, i, axis = 1)
    clf = LogisticRegression(random_state=0,tol=0.001).fit(tmp, y_train)
    fs_train = clf.score(tmp, y_train)
    fs_test = clf.score(tmp1, y_test)
    if fs_train > base_train:
        if fs_test >= base_test:
            print("Feature Selection:")
            print("remove",i)
            print("train Accuracy:",fs_train)
            print("test Accuracy:",fs_test,"\n")


# In[6]:


# Random Forest Sklearn
cancer = load_breast_cancer()
y = cancer.target
X = cancer.data
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(max_depth=5, random_state=6)
clf = clf.fit(X_train,y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
print("Random Forest:")
print("train Accuracy:",metrics.accuracy_score(y_train, train_pred))
print("test Accuracy:",metrics.accuracy_score(y_test, test_pred),"\n")


# In[7]:


# Random Forest Algorithm from scratch
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)

# load and prepare data
dataset = []
for i in range(len(X)):
    tmp = [X[i][j] for j in range(len(X[0]))]
    tmp.append(y[i])
    dataset.append(tmp)
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 4
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
print("Random Forest:")
for n_trees in [5, 10, 15]:
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# In[ ]:




