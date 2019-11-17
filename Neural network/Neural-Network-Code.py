#!/usr/bin/env python
# coding: utf-8

# # Neural Network from Scratch
# Code modified from https://github.com/adventuresinML/adventures-in-ml-code/blob/master/neural_network_tutorial.py
# 
# The notation in this website is almost the same as the notation we are using in class.  Instead of $a$ the author uses $h$, and instead of $N$, the author uses $m$. (I have modified the code below to use $a$ and $N$.)
# 
# Please read about this implementation starting at page 27 from the website listed above.

# ## The first thing we will do is import all the libraries
# 
# We will be using the lower resolution MINST data set

# In[1]:


from sklearn.datasets import load_digits # The MNIST data set is in scikit learn data set
from sklearn.preprocessing import StandardScaler  # It is important in neural networks to scale the date
from sklearn.model_selection import train_test_split  # The standard - train/test to prevent overfitting and choose hyperparameters
from sklearn.metrics import accuracy_score # 
import numpy as np
import numpy.random as r # We will randomly initialize our weights
import matplotlib.pyplot as plt 


# ## Looking at the data
# 
# After we load the data, we print the shape of the data and a pixelated digit.
# 
# We also show what the features of one example looks like.
# 
# The neural net will learn to estimate which digit these pixels represent.

# In[2]:


digits=load_digits()
X = digits.data
print("The shape of the digits dataset:") 
print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[0])
plt.show()
y = digits.target
print(y[0:1])
print(X[0,:])


# ## 1) Scale the dataset
# The training features range from 0 to 15.  To help the algorithm converge, we will scale the data to have a mean of 0 and unit variance

# In[3]:


X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

X[0,:] # Looking the new features after scaling


# ## 2) Creating training and test datasets
# We split the data into training and test data sets. We will train the neural network with the training dataset, and evaluate our neural network with the test dataset 

# In[4]:


#Split the data into training and test set.  60% training and %40 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# ## 3) Setting up the output layer

# ### One hot encoding
# Our target is an integer in the range [0,..,9], so we will have 10 output neuron's in our network.  
# 
# -  If  $y=0$, we want the output neurons to have the values $(1,0,0,0,0,0,0,0,0,0)$
# 
# -  If  $y=1$ we want the output neurons to have the values $(0,1,0,0,0,0,0,0,0,0)$
# -  etc
# 
# Thus we need to change our target so it is the same as our hoped for output of the neural network.  
# -  If $y=0$ we change it into the vector $(1,0,0,0,0,0,0,0,0,0)$. 
# -  If $y=1$ we change it into the vector $(0,1,0,0,0,0,0,0,0,0)$
# -  etc
# 
# See page 29 from the website listed above
# 
# The code to covert the target vector.

# In[5]:


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


# Converting the training and test targets to vectors 

# In[6]:


# convert digits to vectors
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)


# A quick check to see that our code performs as we expect 

# In[7]:


print(y_train[0:4])
print(y_v_train[0:4])


# ## 4) Creating the neural network

# ### The activation function and its derivative
# 
# We will use the sigmoid activation function:  $f(z)=\frac{1}{1+e^{-z}}$
# 
# The deriviative of the sigmoid function is: $f'(z) = f(z)(1-f(z))$ 

# In[8]:


def f(z, act):
    #ReLU
    if act == 2:
        for i in range(len(z)):
            z[i] = max(0,z[i])
        return z
    #tanh
    elif act == 3:
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    #sigmoid
    else:
        return 1 / (1 + np.exp(-z))
def f_deriv(z, act):
    #ReLU
    if act == 2:
        for i in range(len(z)):
            if z[i] < 0:
                z[i] = 0
            else:
                z[i] = 1
        return z
    #tanh
    elif act == 3:
        return (1 - f(z, act)**2)
    #sigmoid
    else:
        return f(z, act) * (1 - f(z, act))


# ### Creating and initialing W and b
# We want the weights in W to be different so that during back propagation the nodes on a level will have different gradients and thus have different update values.
# 
# We want the  weights to be small values, since the sigmoid is almost "flat" for large inputs.
# 
# Next is the code that assigns each weight a number uniformly drawn from $[0.0, 1.0)$.  The code assumes that the number of neurons in each level is in the python list *nn_structure*.
# 
# In the code, the weights, $W^{(\ell)}$ and $b^{(\ell)}$ are held in a python dictionary

# In[9]:


def setup_and_init_weights(nn_structure, W_ini):
    W = {} #creating a dictionary i.e. a set of key: value pairs
    b = {}
    for l in range(1, len(nn_structure)):
        #Logistic
        if W_ini == 2:
            rr = np.sqrt((6 / (64 + 10)))
            W[l] = r.uniform(-rr, rr, (nn_structure[l], nn_structure[l-1]))
        #Hyperbolic tangent
        elif W_ini == 3:
            rr = 4 * np.sqrt((6 / (64 + 10)))
            W[l] = r.uniform(-rr, rr, (nn_structure[l], nn_structure[l-1]))
        #ReLU
        elif W_ini == 4:
            rr = np.sqrt(2) * np.sqrt((6 / (64 + 10)))
            W[l] = r.uniform(-rr, rr, (nn_structure[l], nn_structure[l-1]))
        #Random
        else:
            W[l] = r.random_sample((nn_structure[l], nn_structure[l-1])) #Return “continuous uniform” random floats in the half-open interval [0.0, 1.0). 
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


# ### Initializing $\triangledown W$ and $\triangledown b$
# Creating $\triangledown W^{(\ell)}$ and $\triangledown b^{(\ell)}$ to have the same size as $W^{(\ell)}$ and $b^{(\ell)}$, and setting $\triangledown W^{(\ell)}$, and  $\triangledown b^{(\ell)}$ to zero

# In[10]:


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


# ## Feed forward
# Perform a forward pass throught the network.  The function returns the values of $a$ and $z$

# In[11]:


def feed_forward(x, W, b, act):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        a[l+1] = f(z[l+1], act) # a^(l+1) = f(z^(l+1))
    return a, z


# ## Compute $\delta$
# The code below compute $\delta^{(s_l)}$ in a function called "calculate_out_layer_delta",  and  computes $\delta^{(\ell)}$ for the hidden layers in the function called "calculate_hidden_delta".  
# 
# If we wanted to have a different cost function, we would change the "calculate_out_layer_delta" function.
# 

# In[12]:


def calculate_out_layer_delta(y, a_out, z_out, act):
    # delta^(nl) = -(y_i - a_i^(nl)) * f'(z_i^(nl))
    return -(y-a_out) * f_deriv(z_out, act) 


def calculate_hidden_delta(delta_plus_1, w_l, z_l, act):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l, act)


# ## The Back Propagation Algorithm
# 

# In[13]:


def train_nn(nn_structure, X, y, iter_num, alpha, act, W_ini):
    W, b = setup_and_init_weights(nn_structure, W_ini)
    cnt = 0
    N = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(N):
            delta = {}
            # perform the feed forward pass and return the stored a and z values, to be used in the
            # gradient descent step
            a, z = feed_forward(X[i, :], W, b, act)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], a[l], z[l], act)
                    avg_cost += np.linalg.norm((y[i,:]-a[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l], act)
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(a^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(a[l][:,np.newaxis]))# np.newaxis increase the number of dimensions
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/N * tri_W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/N * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers, act):
    N = X.shape[0]
    y = np.zeros((N,))
    for i in range(N):
        a, z = feed_forward(X[i, :], W, b, act)
        y[i] = np.argmax(a[n_layers])
    return y


# In[14]:


def train_nn_reg(nn_structure, X, y, iter_num, alpha, lamb, act, W_ini):
    W, b = setup_and_init_weights(nn_structure, W_ini)
    cnt = 0
    N = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(N):
            delta = {}
            # perform the feed forward pass and return the stored a and z values, to be used in the
            # gradient descent step
            a, z = feed_forward(X[i, :], W, b, act)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], a[l], z[l], act)
                    avg_cost += np.linalg.norm((y[i,:]-a[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l], act)
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(a^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(a[l][:,np.newaxis]))# np.newaxis increase the number of dimensions
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/N * tri_W[l] + lamb * W[l])
            b[l] += -alpha * (1.0/N * tri_b[l] + lamb * b[l])
        # complete the average cost calculation
        avg_cost = 1.0/N * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


# ## Running the neural network
# 
# Our code assumes the size of each layer in our network is held in a list.  The input layer will have 64 neurons (one for each pixel in our 8 by 8 pixelated digit).  Our hidden layer has 30 neurons (you can change this value).  The output layer has 10 neurons.
# 
# Next we create the python list to hold the number of neurons for each level and then run the neural network code with our training data.
# 
# This code will take some time...

# In[15]:


nn_structure = [64, 30, 10]
    
# train the NN
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train, 3000, 0.25, 1, 1)


# ### Plotting the learning curve
# 

# In[16]:


# plot the avg_cost_func
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()


# ## 5) Assessing accuracy
# Next we determine what percentage the neural network correctly predicted the handwritten digit correctly on the test set

# In[17]:


# get the prediction accuracy and print
y_pred = predict_y(W, b, X_test, 3, 1)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[18]:


#(a)Add a regularization term to the cost function
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 1, 1)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

y_pred = predict_y(W, b, X_test, 3, 1)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[19]:


#(b)ReLU
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 2, 1)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

y_pred = predict_y(W, b, X_test, 3, 2)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[20]:


#(c)tanh
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 3, 1)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[21]:


#(d)(e)(f)
print('Weight initialization: Random\n')
print('(Alpha, Lambda) = (0.25, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 3, 1)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.25, 0.5)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.5, 3, 1)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.01, 3, 1)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.05)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.05, 3, 1)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.8, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.8, 0.01, 3, 1)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[22]:


print('Weight initialization: Logistic\n')
print('(Alpha, Lambda) = (0.25, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 3, 2)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.25, 0.5)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.5, 3, 2)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.01, 3, 2)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.05)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.05, 3, 2)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.8, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.8, 0.01, 3, 2)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[23]:


print('Weight initialization: Hyperbolic tangent\n')
print('(Alpha, Lambda) = (0.25, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 3, 3)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.25, 0.5)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.5, 3, 3)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.01, 3, 3)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.05)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.05, 3, 3)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.8, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.8, 0.01, 3, 3)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))


# In[24]:


print('Weight initialization: ReLU\n')
print('(Alpha, Lambda) = (0.25, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.01, 3, 4)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.25, 0.5)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.25, 0.5, 3, 4)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.01, 3, 4)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.5, 0.05)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.5, 0.05, 3, 4)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
print('\n')

print('(Alpha, Lambda) = (0.8, 0.01)')
W, b, avg_cost_func = train_nn_reg(nn_structure, X_train, y_v_train, 3000, 0.8, 0.01, 3, 4)
y_pred = predict_y(W, b, X_test, 3, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

