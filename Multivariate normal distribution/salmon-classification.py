#!/usr/bin/env python
# coding: utf-8

# # Salmon classification with the bivariate Gaussian
# 
# In this assigment, you will predict if a fish is an 'Alaskan' salmon or a 'Canadian' salmon.
# 
# The algorithm you will use a generative algorithm.  Where you model each class as a **bivariate Gaussian**.

# ## Step 0. Import statements
# 
# The Python programming language, as most programming languages, is augmented by **modules**.  These modules contain functions and classes for specialized tasks needed in machine learning.
# 
# Below, we will `import` three modules:
# * **pandas**
# * **numpy**
# * **matplotlib.pyplot**
# 
# Note that we imported these modules using **aliases**

# In[1]:


# Standard libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np # for better array operations
import matplotlib.pyplot as plt # plotting utilities 

# Module computing the Gaussian density
from scipy.stats import norm, multivariate_normal 


# ## Step 1. Data preparation: loading, understanding and cleaning the dataset

# ### Importing the dataset
# Make sure the file `Salmon_dataset.csv` is in the same directory as this notebook.
# 
# The dataset contains 100  examples, each example has 3 features (*gender, Freshwater, marine*) and a label (*Alaskan, Canadian*).

# In[2]:


# Loading the data set using Panda's in a dataframe 

df = pd.read_csv('Salmon_dataset.csv', delimiter=',') 

#Lets check that everything worked before continuing on
df.head()


# ### Data preprocesssing
# We will change the labels 'Alaskan' and 'Canadian' to $0$ and $1$ respectively.  In our code it is easier to work with numerical values instead of strings.
# 
# Often we will do more dataprepocessing, such as looking for missing values and scaling the data though that is NOT required for this assignment yet. 

# In[3]:


# It is easier to work with the data if the labels are integers
# Changing the 'Origin' column values, map 'Alaskan':0 and 'Canadian':1
df['Origin']=df.Origin.map({'Alaskan':0, 'Canadian':1})

#Lets check that everything worked before continuing on
df.head()


# In[4]:


# We will store the dataframe as a Numpy array
data = df.to_numpy() 

# Split the examples into a training set (trainx, trainy) and test set (testx, testy) 

########## TO DO ##########
n =  data.shape[0] # the number of rows
train_n = int(.9*n) # this test set is a bit small to really evaluate our hypothesis - what could we do to get a better estimate and still keep most of the data to estimate our parameters?
np.random.seed(0) # Our code randomly chooses which examples will be the training data, but for grading purposes we want the random numbers used to seperate the data are the same for everyone
perm = np.random.permutation(n)
trainx = data[perm[0:train_n],1:3] #selecting the two of the features `Freshwater' and 'Marine'
trainy = data[perm[0:train_n],3]
testx = data[perm[train_n:n], 1:3] # We won't look at the testx data until it is time to evauate our hypothesis.  This numpy array contains the set of test data for the assignment
testy = data[perm[train_n:n],3]

##########


# ### Plotting the dataset
# Visualization can be helpful when exploring and getting to know a dataset.

# In[5]:


# plotting the Alaskan salmon as a green dot
plt.plot(trainx[trainy==0,0], trainx[trainy==0,1], marker='o', ls='None', c='g')
# plotting the Canadian salmon as a red dot
plt.plot(trainx[trainy==1,0], trainx[trainy==1,1], marker='o', ls='None', c='r')


# ## Step 2. Model training: implementing Gaussian Discriminant Analysis
# 
# 
# 

# ###  Sufficient statistics
# 
# Just as if we were doing these calculations by hand, we break the process down into managable pieces
# 
# Our first helper function will find the mean and covariance of the Gaussian for a set of examples

# In[6]:


# Input: a design matrix
# Output: a numpy array containing the means for each feature, and a 2-dimensional numpy array containng the covariance matrix sigma

x0 = trainx[trainy==0]
y0 = trainy[trainy==0]

x1 = trainx[trainy==1]
y1 = trainy[trainy==1]

def get_mu(x):
    mu = np.mean(x, axis=0)
    mu = mu.reshape(2, 1)
    return mu

def get_sigma(x):
    sigma = np.cov(x.T)
    return sigma

file1 = open("Alaskan.txt","w")
ans1=np.array_str(get_mu(x0))
ans2=np.array_str(get_sigma(x0))
file1.write("mu0:\n" + ans1 + "\n\n" + "sigma0:\n" + ans2)
file1.close()

file2 = open("Canadian.txt","w")
ans3=np.array_str(get_mu(x1))
ans4=np.array_str(get_sigma(x1))
file2.write("mu1:\n" + ans3 + "\n\n" + "sigma1:\n" + ans4)
file2.close()

def pdf(x):
    part1 = 1/((2*np.pi)*(np.sqrt(np.linalg.det(get_sigma(x0)))))
    part2 = np.exp(((-1/2)*(x.reshape(2, 1)-(get_mu(x0))).T.dot(np.linalg.inv(get_sigma(x0))).dot(x.reshape(2, 1)-(get_mu(x0)))))
    result1 = part1 * part2
    part3 = 1/((2*np.pi)*(np.sqrt(np.linalg.det(get_sigma(x1)))))
    part4 = np.exp(((-1/2)*(x.reshape(2, 1)-(get_mu(x1))).T.dot(np.linalg.inv(get_sigma(x1))).dot(x.reshape(2, 1)-(get_mu(x1)))))
    result2 = part3 * part4
    if result1 > result2:
        return 0
    else:
        return 1

file3 = open("result.txt","w")    
for x in testx:
    ans5=pdf(x)
    file3.write(str(ans5) + "\n")
file3.close()


# 
# Before moving on, test your code to make sure it works correctly.  
# 
# 
# 

# ### Write the rest of your code here

# In[7]:


"""
x0 = trainx[trainy==0]
y0 = trainy[trainy==0]

x1 = trainx[trainy==1]
y1 = trainy[trainy==1]

num_Canadian = np.sum(trainy[trainy==1])
num_Alaskan = train_n-num_Canadian

def get_mu0():
# Calculate mean for class 0
    x0_sum = x0.sum(axis=0)    
    mu0 = x0_sum / num_Alaskan
    mu0 = mu0.reshape(2, 1)
    print(mu0)
    return mu0

def get_mu1():
# Calculate mean for class 1
    x1_sum = x1.sum(axis=0)    
    mu1 = x1_sum / num_Canadian
    mu1 = mu1.reshape(2, 1)
    print(mu1)
    return mu1

mu0 = get_mu0()
mu1 = get_mu1()
"""

