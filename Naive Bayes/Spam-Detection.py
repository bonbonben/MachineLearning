#!/usr/bin/env python
# coding: utf-8

# # Using Naive Bayes algorithm for spam detection
# 
# In this assigment, you will predict if a sms message is 'spam' or 'ham' (i.e. not 'spam') using the Bernoulli Naive Bayes *classifier*.
# 
# The training data is from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection.  Please download the zip file before running the code below. 
# 

# ## Step 1:  Getting, understanding, and cleaning the dataset
# 

# ###  Importing the dataset

# In[1]:


# Import the usual libraries
import numpy as np 
import pandas as pd  # To read in the dataset we will use the Panda's library
df = pd.read_table('SMSSpamCollection', sep = '\t', header=None, names=['label', 'sms_message'])

# Next we observe the first 5 rows of the data to ensure everything was read correctly
df.head() 


# ### Data preprocesssing
# It would be more convenient if the labels were binary instead of 'ham' and 'spam'.  This way our code can always work with numerical values instead of strings.

# In[2]:


df['label']=df.label.map({'spam':1, 'ham':0})
df.head() # Again, lets observe the first 5 rows to make sure everything worked before we continue


# ### Splitting the dcoument into training set and test set

# In[3]:


# This time we will use sklearn's method for seperating the data
from sklearn.model_selection import train_test_split

df_train_msgs, df_test_msgs, df_ytrain, df_ytest = train_test_split(df['sms_message'],df['label'], random_state=0)

#Looking at the train/test split
print("The number of training examples: ", df_train_msgs.shape[0])
print("The number of test exampels: ", df_test_msgs.shape)

print("The first four labels")
print(df_ytrain[0:4])

print("The first four sms messages")
print(df_train_msgs[0:4])


# ###  Creating the feature vector from the text (feature extraction)
# 
# Each message will have its own feature vector.  For each message we will create its feature vector as we discussed in class; we will have a feature for every word in our vocabulary.  The $j$th feature is set to one ($x_j=1$) if the $j$th word from our vocabulary occurs in the message, and set the $j$ feature is set to $0$ otherwise ($x_j=0$).
# 
# We will use the sklearn method CountVectorize to create the feature vectors for every messge.
# 
# We could have written the code to do this directly by performing the following steps:
# * remove capitalization
# * remove punctuation 
# * tokenize (i.e. split the document into individual words)
# * count frequencies of each token 
# * remove 'stop words' (these are words that will not help us predict since they occur in most documents, e.g. 'a', 'and', 'the', 'him', 'is' ...

# In[4]:


# importing the library
from sklearn.feature_extraction.text import CountVectorizer
# creating an instance of CountVectorizer
# Note there are issues with the way CountVectorizer removes stop words.  To learn more: https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
vectorizer = CountVectorizer(binary = True, stop_words='english')

# If we wanted to perform Multnomial Naive Bayes, we would include the word counts and use the following code
#vectorizer = CountVectorizer(stop_words='english')

# To see the 'count_vector' object
print(vectorizer)


# In[5]:


# To see the 'stop words' 
#print(vectorizer.get_stop_words())


# In[6]:


# Create the vocabulary for our feature transformation
vectorizer.fit(df_train_msgs)

# Next we create the feature vectors for both the training data and the test data
X_train = vectorizer.transform(df_train_msgs).toarray() # code to turn the training emails into a feature vector
X_test = vectorizer.transform(df_test_msgs).toarray() # code to turn the test email into a feature vector

# Changing the target vectors data type  
y_train=df_ytrain.to_numpy() # Convereting from a Panda series to a numpy array
y_test = df_ytest.to_numpy()

# To observe what the data looks like 
#print("The label of the first training example: ", y_train[0])
#print("The first training example: ", X_train[0].tolist())# I needed to covernt the datatype to list so all the feature values would be printed


# # Your code goes here

# In[7]:


# You should NOT use the features of sklearn library in your code.
#### TO-DO #####
import numpy as np
import math

#print(len(X_train))
#print(y_train[1])

m = 0.01
spam = np.zeros(len(X_train[0]))
ham = np.zeros(len(X_train[0]))
totalspam = sum(y_train)
for i in range(0,len(X_train)):
    for j in range(0,len(X_train[0])):
        if X_train[i][j] == 1 and y_train[i] == 1:
            spam[j] += 1
        elif X_train[i][j] == 1 and y_train[i] == 0:
            ham[j] += 1
spam = (spam + m) / (totalspam + 2*m)
ham = (ham + m ) / (len(X_train)-totalspam + 2*m)
#print(spam.tolist())
#for i in range(0,len(spam)):
#    if spam[i] == 0:
#        spam[i] = math.pow(10, -10)
#    if ham[i] == 0:
#        ham[i] = math.pow(10, -10)

##############


# In[8]:


tspam = sum(y_train)/len(y_train)
tham = 1 - tspam

file1 = open("P(y).txt","w")
file1.write("P(y=1): "+ str(tspam) +"\n" + "P(y=0): "+ str(tham) +"\n")
file1.close()

tspam = math.log(tspam)
tham = math.log(tham)
#print(tspam, tham)


# In[9]:


#print(X_test[0].tolist())
pSpam = tspam * np.ones(len(X_test))
pHam = tham * np.ones(len(X_test))
ans = []
for j in range(0,len(X_test)):
    for i in range(0,len(X_test[0])):
        if X_test[j][i] == 1:
            pSpam[j] += math.log(spam[i])
            pHam[j] += math.log(ham[i])
        else:
            #if 1-spam[i] != 0:
                pSpam[j] += math.log(1-spam[i])
            #else:
            #    pSpam[j] += math.log(math.pow(10, -10))
            #if 1-ham[i] != 0:
                pHam[j] += math.log(1-ham[i])
            #else:
            #    pHam[j] += math.log(math.pow(10, -10))

    if pSpam[j] >= pHam[j]:
        ans.append(1)
    else:
        ans.append(0)


# In[10]:


count = 0
for i in range(0,len(ans)):
    if ans[i] == y_test[i]:
        count += 1
#print (count/len(ans))


# In[11]:


file2 = open("p(xi y=1).txt","w")
file2.write("p(xi|y=1): \n")
file2.write(str(spam.tolist()))
file2.close()
file3 = open("p(xi y=0).txt","w")
file3.write("p(xi|y=0): \n")
file3.write(str(ham.tolist()))
file3.close()

file4 = open("50 predicted.txt","w")
file4.write(str(ans[:50]))
file4.close()

file5 = open("examples classified.txt","w")
file5.write("correct: "+ str(count) +"\n")
file5.write("incorrect: "+ str(len(ans)-count) +"\n")
file5.write("error: " + str(1-count/len(ans)) + "\n")
file5.close()


# In[ ]:




