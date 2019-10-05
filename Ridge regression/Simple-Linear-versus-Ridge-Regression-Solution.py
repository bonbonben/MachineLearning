#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment 3 - Simple Linear versus Ridge Regression 

# Your friend Bob just moved to Boston. He is a real estate agent who is trying to evaluate the prices of houses in the Boston area. He has been using a linear regression model but he wonders if he can improve his accuracy on predicting the prices for new houses. He comes to you for help as he knows that you're an expert in machine learning. 
# 
# As a pro, you suggest doing a *polynomial transformation*  to create a more flexible model, and performing ridge regression since having so many features compared to data points increases the variance. 
# 
# Bob, however, being a skeptic isn't convinced. He wants you to write a program that illustrates the difference in training and test costs for both linear and ridge regression on the same dataset. Being a good friend, you oblige and hence this assignment :) 

# In this notebook, you are to explore the effects of ridge regression.  We will use a dataset that is part of the sklearn.dataset package.  Learn more at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

# ## Step 1:  Getting, understanding, and preprocessing the dataset
# 
# We first import the standard libaries and some libraries that will help us scale the data and perform some "feature engineering" by transforming the data into $\Phi_2({\bf x})$

# In[1]:


import numpy as np
import sklearn
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.model_selection import KFold


# ###  Importing the dataset

# In[2]:


# Import the boston dataset from sklearn
boston_data = load_boston()


# In[3]:


#  Create X and Y variables - X holding the .data and Y holding .target 
X = boston_data.data
y = boston_data.target

#  Reshape Y to be a rank 2 matrix 
y = y.reshape(X.shape[0], 1)

# Observe the number of features and the number of labels
print('The number of features is: ', X.shape[1])
# Printing out the features
print('The features: ', boston_data.feature_names)
# The number of examples
print('The number of exampels in our dataset: ', X.shape[0])
#Observing the first 2 rows of the data
print(X[0:2])


# We will also create polynomial feeatures for the dataset to test linear and ridge regression on data with d = 1 and data with d = 2. Feel free to increase the # of degress and see what effect it has on the training and test error. 

# In[4]:


# Create a PolynomialFeatures object with degree = 2. 
# Transform X and save it into X_2. Simply copy Y into Y_2 
poly = PolynomialFeatures(degree=2)
X_2 = poly.fit_transform(X)
y_2 = y


# In[5]:


# the shape of X_2 and Y_2 - should be (506, 105) and (506, 1) respectively
print(X_2.shape)
print(y_2.shape)


# # Your code goes here

# In[6]:


# TODO - Define the get_coeff_ridge_normaleq function. Use the normal equation method.
# TODO - Return w values

def get_coeff_ridge_normaleq(X_train, y_train, alpha):
    # use np.linalg.pinv(a)
    #### TO-DO #####
    x_T = np.transpose(X_train)
    lamb = np.identity(len(X_train[0]))*alpha
    w = np.linalg.pinv(np.mat(x_T) * np.mat(X_train) + lamb) * x_T * y_train
    
    ##############
    return w


# In[7]:


# TODO - Define the evaluate_err_ridge function.
# TODO - Return the train_error and test_error values


def evaluate_err(X_train, X_test, y_train, y_test, w, alpha = 0): 
    #### TO-DO #####
    train = np.square(X_train * w - y_train)
    temp1 = 0
    for i in range(0,len(train)):
        temp1 += train[i]
    
    test = np.square(X_test * w - y_test)
    temp2 = 0
    for i in range(0,len(test)):
        temp2 += test[i]
    
    ridge = alpha * sum(np.square(w))
    train_error = temp1/len(train) + ridge
    test_error = temp2/len(test) + ridge
    ##############
    return train_error, test_error


# In[8]:


#w = get_coeff_ridge_normaleq(X, y, 10)
#evaluate_err(X, X, y, y, w, 10)


# In[9]:


# TODO - Finish writting the k_fold_cross_validation function. 
# TODO - Returns the average training error and average test error from the k-fold cross validation
# use Sklearns K-Folds cross-validator: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

def k_fold_cross_validation(k, X, y, alpha):
    kf = KFold(n_splits=2, random_state=21, shuffle=True)
    total_E_val_test = 0
    total_E_val_train = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # centering the data so we do not need the intercept term (we could have also chose w_0=average y value)
        y_train_mean = np.mean(y_train)
        y_train = y_train - y_train_mean
        y_test = y_test - y_train_mean
        # scaling the data matrix
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
    # determine the training error and the test error
    #### TO-DO #####
    w = get_coeff_ridge_normaleq(X_train, y_train, alpha)
    train_error, test_error = evaluate_err(X_train, X_test, y_train, y_test, w, alpha)
    total_E_val_train += train_error
    total_E_val_test += test_error
    E_val_test = total_E_val_test / len(X_test)
    E_val_train = total_E_val_train / len(X_train)
    ##############
    return  E_val_test, E_val_train
    


# In[10]:


alpha = np.logspace(1, 7, num=13)
print("ridge regression:")
print("alpha: ", "0")
E_val_test, E_val_train = k_fold_cross_validation(10, X, y, 0)
print(E_val_test, E_val_train)
for i in range(0,len(alpha)):
    print("alpha: ", alpha[i])
    E_val_test, E_val_train = k_fold_cross_validation(10, X, y, alpha[i])
    print(E_val_test, E_val_train)


# In[11]:


alpha = np.logspace(1, 7, num=13)
print("polynomial transformation:")
print("alpha: ", "0")
E_val_test, E_val_train = k_fold_cross_validation(10, X_2, y_2, 0)
print(E_val_test, E_val_train)
for i in range(0,len(alpha)):
    print("alpha: ",alpha[i])
    E_val_test, E_val_train = k_fold_cross_validation(10, X_2, y_2, alpha[i])
    print(E_val_test, E_val_train)


# In[12]:


def predict(X, y, X_pre,alpha):    

    X_train, X_test = X, X_pre
    y_train, y_test = y, [] 
        
    # centering the data so we do not need the intercept term (we could have also chose w_0=average y value)
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean
    #y_test = y_test - y_train_mean
    # scaling the data matrix
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_train)
        
    # determine the training error and the test error
    #### TO-DO #####
    w = get_coeff_ridge_normaleq(X_train, y_train, alpha)
    ##############

    y = X_pre * w
    y = y + y_train_mean
    print(y)


# In[13]:


X_pre=[[5, 0.5, 2, 0, 4, 8, 4, 6, 2, 2, 2, 4, 5.5]]
X_pre = poly.fit_transform(X_pre)
predict(X_2, y_2, X_pre, 0)


# In[14]:


X_pre=[[5, 0.5, 2, 0, 4, 8, 4, 6, 2, 2, 2, 4, 5.5]]
predict(X, y, X_pre, 0)

