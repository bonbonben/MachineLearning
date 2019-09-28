#!/usr/bin/env python
# coding: utf-8

# # Using the K-NN algorithm for classification of iris
# 
# In this assigment, you will classify if an Iris is 'Iris Setosa' or 'Iris Versicolour' or 'Iris Virginica' using the k nearest neighbor algorithm.
# 
# The training data is from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/iris.  Please download the dataset before running the code below. 

# ## Step 1:  Getting, understanding, and cleaning the dataset
# 

# ###  Importing the dataset
# 

# In[1]:


# Import the usual libraries
import matplotlib.pyplot as plt # plotting utilities 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd  # To read in the dataset we will use the Panda's library
df = pd.read_csv('iris.csv', header=None, names = ["sepal length[cm]","sepal width[cm]","petal length[cm]", "petal width", "label"])

# Next we observe the first 5 rows of the data to ensure everything was read correctly
df.head()


# ### Data preprocesssing
# It would be more convenient if the labels were integers instead of 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'.  This way our code can always work with numerical values instead of strings.

# In[2]:


df['label'] = df.label.map({'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2})
df.head()# Again, lets observe the first 5 rows to make sure everything worked before we continue


# In[3]:


# This time we will use sklearn's method for seperating the data
from sklearn.model_selection import train_test_split
names = ["sepal length[cm]","petal width"]
#After completing the assignment, try your code with all the features
#names = ["sepal length[cm]","sepal width[cm]","petal length[cm]", "petal width"]
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df[names],df['label'], random_state=0)

X_train=df_X_train.to_numpy()
X_test=df_X_test.to_numpy()
y_train=df_y_train.to_numpy()
y_test=df_y_test.to_numpy()

#Looking at the train/test split
print("The number of training examples: ", X_train.shape[0])
print("The number of test exampels: ", X_test.shape[0])

print("The first four training labels")
print(y_train[0:4])

print("The first four iris' measurements")
print(X_test[0:4])


# ## visualizing the data set
# 
# Using a scatter plot to visualize the dataset

# In[4]:


iris_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
for i in range(0,3):
    plt.scatter(X_train[y_train == i, 0],
                X_train[y_train == i, 1],
            marker='o',
            label='class '+ str(i)+ ' '+ iris_names[i])

plt.xlabel('sepal width[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='lower right')

plt.show()


# # Your code goes here

# In[5]:


#from scipy.spatial import distance
def euclidean_distance(x1, x2):
  #### TO-DO #####      
        return np.linalg.norm(a-b)
        #return distance.euclidean(a, b)
  ##############


# In[6]:


from scipy.spatial import distance
from collections import Counter

class kNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbors(self, X_test, k):
        neighbors = []
        for row in X_test:
            label = self.closest(row,k)
            neighbors.append(label)
        return neighbors

    def closest(self, row, k):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((i,distance.euclidean(row,self.X_train[i])))
        distances = sorted(distances, key=lambda x:x[1])[0:k]
        k_indeces = []
        for i in range(k):
            k_indeces.append(distances[i][0])
        k_labels = []
        for i in range(k):
            k_labels.append(self.y_train[k_indeces[i]])
        c = Counter(k_labels)
        return c.most_common()[0][0]

from sklearn.neighbors import KNeighborsClassifier
classifier = kNN()
classifier.fit(X_train, y_train)
ans1 = classifier.get_neighbors(X_test, 1)
ans2 = classifier.get_neighbors(X_test, 3)
ans3 = classifier.get_neighbors(X_test, 5)

from sklearn.metrics import accuracy_score

#print(y_train)
count0=0
count1=0
count2=0
ans4=0
for i in range(0,len(y_train)):
    if y_train[i] == 0:
        count0+=1
    if y_train[i] == 1:
        count1+=1
    if y_train[i] == 2:
        count2+=1
#print(count0,count1,count2)
result=max(count0,count1,count2)
#print(result)
if result == count0:
    for i in range(0,len(y_test)):
        if y_test[i] == 0:
            ans4+=1
if result == count1:
    for i in range(0,len(y_test)):
        if y_test[i] == 1:
            ans4+=1
if result == count2:
    for i in range(0,len(y_test)):
        if y_test[i] == 2:
            ans4+=1

#print(y_test)
print(ans1)
print ("accuracy for k = 1:", accuracy_score(y_test, ans1)*100, "%")
print(ans2)
print ("accuracy for k = 3:", accuracy_score(y_test, ans2)*100, "%")
print(ans3)
print ("accuracy for k = 5:", accuracy_score(y_test, ans3)*100, "%")
print ("accuracy for Zero-R:", (ans4/len(y_test))*100, "%")


# In[7]:


#### TO-DO ##### 


# In[ ]:




