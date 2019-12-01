#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines (SVM) - Hard Margin Case
# ---
# Recall the binary classification problem that SVMs try to solve. The hard margin SVM finds the seperating decision bounary with the largest margin. 
# 
# <img src="MaxMargin.jpg" height="400" width="400">
# 
# In this assigment, you will write the code to find the maximum margin for the hard margin case.
# 
# We showed in class that we could solve this problem  by reducing it to the problem of solving a quadratic programming problem (QP).  There are many solvers for quadratic programming problems. We will use the *Convex Optimization Library*, [CVXOPT](https://cvxopt.org/userguide/coneprog.html#quadratic-programming); a free software package that works well with Python and numpy. 
# You will need to install [CVXOPT](https://cvxopt.org/install/)
# 
# In CVXOPT, the quadratic programming problem solver, <b><i>cvxopt.solvers.qp</i></b>,  solves the following problem:
# 
# $$\begin{eqnarray} \min_{x}\frac{1}{2} x^{T}Px - q^{T}x \nonumber \\\ \textrm{s.t.}\quad Gx \preceq h \\\ \textrm{and}\quad Ax = b \end{eqnarray}$$
# 
#  Note that $ Gx \preceq h $ is taken elementwise. 
# 
# The solver's (simplified) API is `cvxopt.solvers.qp(P, q, G, h, A, b)` 
# where only $P$ and $q$ are required. 
# 
# You will need to match the solver's API.
# 
# The solver's argument's type must be CVXOPT matrices.  Please look at this [link](https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf) for more information. I suggest you first create the arguments as NumpPy arrays and matrices and then convert them to CVXOPT matrices (For example, first import the library: `from cvxopt import matrix` then convert a NumPy matrix `P` to a CVXOPT matrix using ` P = matrix(P)`)
# 
# What is return by the solver  is a Python dictionary.  If you save the return value in a variable called `sol` (i.e. `sol = solvers.qp(...)`), you can access to the solution of the quadratic programming problem by typing `sol["x"]`.

# # Hard Margin Case

#  We will use synthetic data. The following code will reads the linearly sepearable data and plots the points.
# 

# In[1]:


import numpy as np
import csv
from numpy import genfromtxt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


X = genfromtxt('X.csv', delimiter=',') # reading in the data matrix
y = genfromtxt('y.csv', delimiter=',') # reading in the labels
idx_1 = np.where(y == 1)
idx_0 = np.where(y == -1)

plt.figure(figsize=(15,9))
plt.scatter(X[idx_1,0], X[idx_1,1], s=30, c='b', marker="o")
plt.scatter(X[idx_0,0], X[idx_0,1], s=30, c='r', marker="o")
plt.xlabel('x1')
plt.ylabel('x2');

plt.show()


# ## SVM Primal  Problem
# ---
# ### Problem 1.1
# 
# You will now use the quadratic problem solver to find the maximum margin for the SVM primal problem:
# 
# $$\begin{eqnarray}\left.\begin{aligned}  
# &\min_{w}\frac{1}{2}||w||^{2} & \nonumber \\\ 
# &\textrm{s.t.}  \quad y^{(i)}(w^{T}{\bf x}^{(i)} + w_0) \ge 1 \quad \forall i \end{aligned}\right.\end{eqnarray}$$
# 
# 
# Look back at the lecture notes for the primal problem.  Please note that the variable names used in the lecture are different from the variable names given to decribe the API for CVXOPT's quadratic problem solver.
# 

# 
# 
#  Write the function `linear_svm(X, y)` that:
# - takes in as arguments the data matrix $X$ and the labels $\bf y$ 
# - solves the SVM primal QP problem 
# - returns  ${\bf w}$ and $w_0$

# In[2]:


get_ipython().system('pip install cvxopt')
from cvxopt import matrix, solvers
from optparse import OptionParser

# define kernel matrix
def KernelMatrix(X, kernel):
    (N, k) = np.shape(X)
    tmp = np.ndarray((N, N))
    for idx1, x in enumerate(X):
        for idx2, y in enumerate(X):
            tmp[idx1][idx2] = kernel(x, y)
    return tmp  
    
def linear_svm(X, y, C=1, tol=0.0001):
    solvers.options['show_progress'] = False
    ## Write code here ##
    linearKernel = lambda x1, x2: np.dot(x1, x2)    # define the dot product of the feature vectors
    K = KernelMatrix(X, linearKernel)
    (N, k) = np.shape(X)
    
    P = matrix(np.outer(y, y) * K, tc='d')  # tc='d' means double-precision numbers in the matrix
    q = matrix(-1 * np.ones(N), tc='d')
    G = matrix(np.row_stack((-1 * np.diag(np.ones(N)), np.diag(np.ones(N)))), tc='d')
    h = matrix(np.concatenate((np.zeros(N), C*np.ones(N))), tc='d')
    A = matrix(y, tc='d').trans()
    b = matrix(0, tc='d')
    
    sol = solvers.qp(P, q, G, h, A, b)['x']
    sol = np.ravel(sol)    
    svIndices = np.where(sol > tol)[0]
    print(svIndices)
    svData = {'svs': sol[svIndices], 'y': y[svIndices], 'X': X[svIndices]}
    #print(svData)
    weights = np.dot(np.transpose(np.multiply(svData['svs'], svData['y'])), svData['X'])
    #print(weights)
    
    w0 = svData['y'][3]-weights[0]*svData['X'][3][0] - weights[1]*svData['X'][3][1]
    #print(w0)
    
    #print(svData['y'][3],svData['X'][3][0],svData['X'][3][1])
    for i in range(len(svData)):
        print(svData['y'][i]-weights[0]*svData['X'][i][0] - weights[1]*svData['X'][i][1])
    
    weights = np.insert(weights, 0, w0, axis=0)
    
    ##
    return weights

# fit svm classifier
weights = linear_svm(X, y)
w0 = weights[0]
w = weights[1:3]
print(weights)
print(w0)
print(w)


# ### Plotting the  decision boundary

# In[3]:


def plot_data_with_decision_boundary(X, y, w, w0, fig_size=(15, 9), labels=['x1', 'x2']):
    COLORS = ['blue', 'red']
    unique = np.unique(y)
    


    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == -1)

    plt.figure(figsize=(15,9))
    plt.scatter(X[idx_1,0], X[idx_1,1], s=30, c='b', marker="o")
    plt.scatter(X[idx_0,0], X[idx_0,1], s=10, c='r', marker="o")
    plt.xlabel('x1')
    plt.ylabel('x2');
        
    ## Write code here ##
            
    slope = -w[0] / w[1]
    intercept = -w0 / w[1]
    x = np.arange(0, 6)
    plt.plot(x, x * slope + intercept, 'k-')
    
    plt.grid()

# plotting the points and decision boundary   
plot_data_with_decision_boundary(X, y, w, w0)


# ### Problem 1.2 
# Determine which points are closest to the decision boundary. What is the functional margin of the points closest to the decision boundary?   

# In[4]:


# Your code goes here
import math
dist = []
for i in range(len(y)):
    dist.append(abs(w[0]*X[i][0]+ w[1]*X[i][1] + w0)/math.sqrt(w[0]**2+w[1]**2))
print(min(dist))
#for i in range(len(dist)):
#    if abs(dist[i]) < 0.4:
#        print(i,dist[i], y[i])


# ### Problem 1.3
# 
# Write the decision function $f_{\texttt{primal}}({\bf x})$ to predict examples.  Use this function to predict the label of $(3.0, 1.5)^T$ and $(1.2, 3.0)^T$

# In[5]:


def f_primal(x):
    ## your code 
    num = w[0]*x[0]+ w[1]*x[1] + w0
    if num < 0:
        return -1
    else:
        return 1

# write the code to predict (3.0, 1.5)^T and (1.2, 3.0)^T here
print(f_primal([3.0, 1.5]))
print(f_primal([1.2, 3.0]))


# ## SVM Dual Problem
# 
# ---
# 
# To keep things simple, we will use a linear kernel $K({\bf x}^{(i)}, {\bf x}^{(j)})={\bf x}^{(i)T}{\bf x}^{(j)}$.  In the statement of the problem below, I replaced ${\bf x}^{(i)}$ with $\phi({\bf x}^{(i)})$ which is different than the lecture notes.
# 
# ### Problem 2.1
# The SVM dual problem was derived in class by:
# - defining the lagrangian of the Primary problem: $L = \frac{1}{2}||w||^{2} - \sum_{i} \alpha^{(i)} [y^{(i)}(w^{T}\phi({\bf x}^{(i)}) + w_0)-1]$
# - equating it's partial derivatives with respect to $w$ and $w_0$ to zero, and 
# - substituting $w$ back into the lagrangian.
# 
# It resulted in a quadratic programming problem (QP) of the following form:
# 
# $$\begin{eqnarray}\left.\begin{aligned}  
# &\min_{\alpha}\frac{1}{2} \alpha^{T}Q\alpha - 1^{T}\alpha \nonumber \\\ &\textrm{s.t.}\quad \alpha_{i} \ge 0 \quad \forall i \\\ 
# &\:\:\:\:\:\quad y^{T}\alpha = 0 \end{aligned}\right.\end{eqnarray}$$
# 
# 

# 
#  Write the function `kernel_svm(x, y)`that:
# - takes in as arguments: the data matrix $X$, the labels $y$ 
# - solves the SVM dual QP problem using a linear kernel
# - returns:  ${\bf \alpha}$

# In[6]:


def kernel_svm(X, y): 
   ## Write your code here
    num = X.shape[0]
    #dim = X.shape[1]
    K = y[:, None] * X
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    G = matrix(-np.eye(num))
    h = matrix(np.zeros(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

   ## 
    return alphas

# fit svm dual classifier
alphas = kernel_svm(X, y)
print(alphas)


# Treat any $\alpha^{(i)} \le 1/1000$ as $0$.
# 
# ### problem 2.2 
# 
# Write a function `compute_w (X, y, alpha)` that: 
# - takes in as arguments the data matrix $X$, labels ${\bf y}$, and ${\bf \alpha}$ 
# - returns ${\bf w}$ and $w_0$ 
# 
# Compare this ${\bf w}$ and $w_0$ computed by the dual with the ${\bf w}$ and $w_0$ computed by the primal

# In[7]:


def compute_decision_boundary (X, y, alphas):
    ## Write your code here
    # get weights
    w = np.sum(alphas * y[:, None] * X, axis = 0)
    # get bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(X[cond], w)
    w0 = b[0]
    
    ##
    return w, w0



w, w0 = compute_decision_boundary(X, y, alphas)
print(w)
print(w0)


# In[8]:


plot_data_with_decision_boundary(X, y, w, w0)


# ### Problem 2.3 
# 
# Write the code to determine which of the training examples are support vectors.  Write the code to determine the functional margin of the support vectors.

# In[9]:


# Write the code here
dist = []
for i in range(len(y)):
    dist.append(abs(w[0]*X[i][0]+ w[1]*X[i][1] + w0)/math.sqrt(w[0]**2+w[1]**2))
print(min(dist))


# ### Problem 2.4
# 
# Write the decision function $f_{\texttt{dual}}({\bf x})=
# \left(\sum_{i\in I}
# \alpha^{(i)}y^{(i)}
# K({\bf x}^{(i)},{\bf x})
# \right)+w_0$ 
# where $I = \{i\mid \alpha^{(i)}\not = 0\}$.  The kernel function will be the linear kernel.  Use the decision function to predict the class of $(3.0, 1.5)^T$ and $(1.2, 3.0)^T$
# 

# In[10]:


def K(xi, xj):
    return np.dot(xi,xj)

def f_dual(x):
    ## your code 
    num = w[0]*x[0]+ w[1]*x[1] + w0
    if num < 0:
        return -1
    else:
        return 1

# write the code to predict (3.0, 1.5)^T and (1.2, 3.0)^T here
print(f_dual([3.0, 1.5]))
print(f_dual([1.2, 3.0]))


# # Experiment on your own
# Do not turn in this part.  You can try to solve the soft margin case and add different kernels.  Try updating `kernel_svm(X,y)` to `kernel_svm(X,y,K)` where you pass the kernel as an argument.

# In[ ]:





# In[ ]:




