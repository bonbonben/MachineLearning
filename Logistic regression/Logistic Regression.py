#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment 5 - Logistic Regression 
# 
# In the assignment, you will use gradient ascent to find the weights for the logistic regression problem.   
# 
# As an example, we will use the widely-used breast cancer data set.  This data set is described here:
# 
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin
# 
# Each sample is a collection of features that were manually recorded by a physician upon inspecting a sample of cells from fine needle aspiration.  The goal is to detect if the cells are benign or malignant.

# ## Step 1:  Getting, preprocessing, and understanding the dataset

# ### Importing the standard libraries

# In[1]:


# Importing important libraries
from sklearn.datasets import load_breast_cancer # taking included data set from Sklearn http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
from sklearn import preprocessing # preprossing is what we do with the data before we run the learning algorithm
from sklearn.model_selection import train_test_split 
import numpy as np
import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing the dataset

# In[2]:


# Loading the dataset
cancer = load_breast_cancer()
y = cancer.target
X = cancer.data


# In[3]:


# Printing the shape of data (X) and target (Y) values 
print(X.shape)
print(y.shape)


# ### Data Pre-Processing
# #### Splitting the data into train and test before scaling the dataset
# 

# In[4]:


# train_test_split to split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y)


# #### Scale the data since we will be using gradient ascent

# In[5]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


# TODO - Print the shape of x_train and y_train 
print(X_train.shape) # It should print (426, 30)
print(y_train.shape) # It should print (426,)


# #### Adding a column of ones to the  matrices $X_{train}$ and  $X_{test}$
# After adding a column of ones $X_{train}=\left[\begin{matrix}
# 1& x^{(1)}_1& x^{(1)}_2 &\ldots& x^{(1)}_d\\
# 1& x^{(2)}_1& x^{(2)}_2 &\ldots& x^{(2)}_d\\
# \vdots & \vdots &\vdots & & \vdots \\
# 1& x^{(N')}_1& x^{(N')}_2 &\ldots& x^{(N')}_d\\
# \end{matrix}\right]$
# 
# Similarly for $X_{test}$

# In[7]:


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


# We can check that everything worked correctly by:
# Printing out the new dimensions
print("The trainng data has dimensions: ", X_train.shape, ". The testing data has dimensions: ",X_test.shape)
# Looking at the first two rows of X_train to check everything worked as expected
print(X_train[0:2])


# ### Understanding the dataset

# In[8]:


# Printing the names of all the features
print(cancer.feature_names)


# In[9]:


# You can add your own code here to better understand the dataset


# 
# # Step 2: Fitting the model
# ## Implementing Logistic Regression Using Gradient Ascent

# 
#  Before writing the gradient ascent code, first write some helpful functions

# 
#  
# ### Sigmoid($z$)
# The first function you will write is sigmoid($z$)
# 
# sigmoid($z$) takes as input a column vector of real numbers, $z^T = [z_1, z_2, ..., z_{N'}]$, where $N'$ is the number of  examples
# 
# It should produce as output a column vector $\left[\frac{1}{1+e^{-z_1}},\frac{1}{1+e^{-z_2}},...,\frac{1}{1+e^{-z_{N'}}}\right]^T$

# In[10]:


# TODO - Write the sigmoid function
def sigmoid(z):
    ## TODO
    
    
    return 1 / (1 + math.exp(-z))
    ## 


# In[11]:


# VERIFY - Sigmoid of 0 should be equal to half
print(sigmoid(0))


# ### Initializing ${\bf w}$
# For testing the next functions, we create a coefficient vector, ${\bf w}$.
# We will initialize the coeffients to be $0$, i.e. ${\bf w}^T = [0,0,\ldots ,0]$ (We could have initialized ${\bf w}$ to any values.)

# In[12]:


# Initialize parameters w
w = np.zeros((X_train.shape[1], 1))
print(w.shape)


# ### Our hypothesis, $h({\bf x})$
# The next  function to write is our hypothesis function. 
# 
# For example if our design matrix $X$ consists of single example $X=[1,x_1,x_2,\ldots,x_d]$ and  weights ${\bf w}^T=[w_0,w_2,\ldots, w_d]$, it returns $h({\bf x})=\frac{1}{1+e^{-\left({w_{0}\cdot 1 +w_1\cdot x_1+\cdots w_d\cdot x_d}\right)}}$
# 
# If given a  matrix consisting of $N'$ examples 
# $X=\left[\begin{matrix}
# 1& x^{(1)}_1& x^{(1)}_2 &\ldots& x^{(1)}_d\\
# 1& x^{(2)}_1& x^{(2)}_2 &\ldots& x^{(2)}_d\\
# \vdots & \vdots &\vdots & & \vdots \\
# 1& x^{(N')}_1& x^{(N')}_2 &\ldots& x^{(N')}_d\\
# \end{matrix}\right]$
# and  weights ${\bf w}^T=[w_0,w_2,\ldots, w_d]$, the function returns a column vector
# $[h({\bf x}^{(1)}),h({\bf x}^{(2)},\ldots, h({\bf x}^{(N')}]^T$

# In[13]:


# predict the probability that a patient has cancer 
# TODO - Write the hypothesis function 
def hypothesis(X, w):
    #TODO
    h = np.ones((X.shape[0],1))
    for i in range(0,X.shape[0]):
        z = np.matmul(X[i], w)
        h[i] = sigmoid(z)
    return h
    ##


# Before moving on, do a quick check that your function can accpet a matrix as an argument. 

# In[14]:


# Compute y_hat using our training examples and w (w is still set to zero).  
# This is just a preliminary test of the hypotheis function
yhat = hypothesis(X_train, w)

# print the sizes of yhat and y as a first check that the function performed correctly
print(yhat.shape) # this should return (426, 1)
print(y_train.shape) # this should return (426,)


# ### Log-Likelihood Function.
# Write the code to calculate the log likelihood function $\ell({\bf w})=
# \sum_{i=1}^{N'}y^{(i)}\ln(h({\bf x}^{(i)})) +(1- y^{(i)})\ln(1-h({\bf x}^{(i)}))$
# 
# The input is a matrix consisting of $N'$ examples $X=\left[\begin{matrix}
# 1& x^{(1)}_1& x^{(1)}_2 &\ldots& x^{(1)}_d\\
# 1& x^{(2)}_1& x^{(2)}_2 &\ldots& x^{(2)}_d\\
# \vdots & \vdots &\vdots & & \vdots \\
# 1& x^{(N')}_1& x^{(N')}_2 &\ldots& x^{(N')}_d\\
# \end{matrix}\right]$
# and a column vector ${\bf y}^T=[y^{(1)},y^{(2)},\dots,y^{(N')}]$ of labels for $X$.
# 
# The output is $\ell({\bf w})$

# In[15]:


# TODO - Write the log likelihood function 
def log_likelihood(X, y, w):
    ##TODO
    log_likelihood = 0
    for i in range(0,X.shape[0]):
        z = np.matmul(X[i], w)
        log_likelihood += y[i] * np.log(sigmoid(z)) + (1 - y[i]) * np.log((1 - sigmoid(z)))    
    ##
    return log_likelihood


# Before moving on, do a quick check of your log_likelihood funciotn

# In[16]:


# VERIFY - The value should be equal to -295.2806989185367.
print(log_likelihood(X_train,y_train,w))


# # Gradient Ascent
# Now write the code to perform gradient ascent.  You will use the update rule from the lecture notes.

# In[17]:


# TODO - Write the gradient ascent function 
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


# ### After completing the code above, run the following

# In[18]:


learning_rate = 0.5
num_iters = 5000 # The number of iteratins to run the gradient ascent algorithm

w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_train, y_train, learning_rate, num_iters)
print(w)
print(log_likelihood_values)


# # Plotting Likelihood v/s Number of Iterations.

# In[19]:


# Run this cell to plot Likelihood v/s Number of Iterations.
iters = np.array(range(0,num_iters,100))
plt.plot(iters,log_likelihood_values,'.-',color='green')
plt.xlabel('Number of iterations')
plt.ylabel('Likelihood')
plt.title("Likelihood vs Number of Iterations.")
plt.grid()


# You should see the likelihood increasing as number of Iterations increase.

# ### The rest of your code goes here

# In[20]:


w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.5, 5000)
print(w)
print(log_likelihood_values)


# In[21]:


# Run this cell to plot Likelihood v/s Number of Iterations.
iters = np.array(range(0,num_iters,100))
plt.plot(iters,log_likelihood_values,'.-',color='green')
plt.xlabel('Number of iterations')
plt.ylabel('Likelihood')
plt.title("Likelihood vs Number of Iterations.")
plt.grid()


# In[22]:


def predict(X_test, w):
    pred = hypothesis(X_test, w)
    res = []
    for i in range(0,len(pred)):
        if pred[i] >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

res = predict(X_test, w)
print(res)
print(y_test)


# In[23]:


def confusion_matrix(res, y_test):
    tp, fn, fp, tn = 0,0,0,0
    for i in range(0,len(res)):
        if res[i] == y_test[i]:
            if res[i] == 1:
                tp += 1
            elif res[i] == 0:
                tn += 1
        else:
            if res[i] == 1:
                fp += 1
            else:
                fn += 1   
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    confusion = [[tp,fn],[fp,tn]]
    return precision, recall, f1, confusion

ans = confusion_matrix(res, y_test)
print(ans)


# # Step 3: Evaluating your model

# In[24]:


for i in range(500, 5000, 500):
    w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.1, i)
    res = predict(X_test, w)
    ans = confusion_matrix(res, y_test)
    print("learning rate: 0.1","number of iterations:", i,"f1 score:", ans[2])


# In[25]:


for i in range(500, 5000, 500):
    w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.2, i)
    res = predict(X_test, w)
    ans = confusion_matrix(res, y_test)
    print("learning rate: 0.2","number of iterations:", i,"f1 score:", ans[2])


# In[26]:


for i in range(500, 5000, 500):
    w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.3, i)
    res = predict(X_test, w)
    ans = confusion_matrix(res, y_test)
    print("learning rate: 0.3","number of iterations:", i,"f1 score:", ans[2])


# In[27]:


for i in range(500, 5000, 500):
    w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.4, i)
    res = predict(X_test, w)
    ans = confusion_matrix(res, y_test)
    print("learning rate: 0.4","number of iterations:", i,"f1 score:", ans[2])


# In[28]:


for i in range(500, 5000, 500):
    w, log_likelihood_values = Logistic_Regresion_Gradient_Ascent(X_test, y_test, 0.5, i)
    res = predict(X_test, w)
    ans = confusion_matrix(res, y_test)
    print("learning rate: 0.5","number of iterations:", i,"f1 score:", ans[2])

