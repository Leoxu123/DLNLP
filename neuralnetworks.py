#!/usr/env/bin python
# -*- coding: utf-8 -*-

'''
    in this part,you're goint to implement
    1)A sigmoid activation function and its gradient
    2)A forward propagation for a simple neural network with cross-entropy cost
    3)A backward propagation algorithm to compute gradients for the parameters
    4)Gradient / derivative check
'''

import numpy as np
import random

#----------------------------------------------
def sigmoid(x):
    """Sigmoid function"""
    x=1/(1+np.exp(-x))
    return x
#----------------------------------------------

#----------------------------------------------
def sigmoid_grad(f):
    """Sigmoid gradient function"""
    f=f*(1-f)
    return f
#----------------------------------------------


#----------------------------------------------
def gradcheck_naive(f,x):
  """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
  """

  rndstate = random.getstate()
  random.setstate(rndstate)
  fx,grad=f(x)
  h=1e-4

  it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
  while not it.finished:
      ix = it.multi_index

      """
        use f(x+h)-f(x-h)/2h to estimate the derivates of parameters in x
        in my practice,f(x+h)-f(x-h)/2h has a higher accuracy than f(x+h)-f(x)/h
      """
      old_value=x[ix]
      x[ix]+=h
      random.setstate(rndstate)
      fx1,notused = f(x)
      x[ix] = old_value-h
      random.setstate(rndstate)
      fx2,notused = f(x)
      numgrad = (fx1-fx2)/(2*h)
      x[ix] = old_value

      reldiff = abs(numgrad-grad[ix])/max(1,abs(numgrad),abs(grad[ix]))
      if reldiff > 1e-5:
          print "Gradient check failed"
          print "First gradient error found at index %s" %str(ix)
          print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
          return
      it.iternext()

  print "Gradient check passed!"
#----------------------------------------------
N = 20
dimensions=[10,5,10]
data = np.random.randn(N,dimensions[0])
labels = np.zeros((N,dimensions[2]))
for i in xrange(N):
    labels[i,random.randint(0,dimensions[2]-1)]=1

params = np.random.randn(((dimensions[0]+1)*dimensions[1]+(dimensions[1]+1)*dimensions[2]))

#----------------------------------------------
def forward_backward_prop(data, labels, params):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    
    ### Unpack network parameters 
    t = 0
    W1 = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))
    
    ### forward propagation
    
    z1=data.dot(W1)+b1
    a1=sigmoid(z1)
    z2=a1.dot(W2)+b2
    a2=sigmoid(z2)
    costPositive = -labels*np.log(a2)
    costNegative = (1-labels)*np.log(1-a2)
    cost = np.sum(costPositive - costNegative)/N
    
    
    ### backward propagation
    gradW1,gradW2,gradb1,gradb2 = 0,0,0,0
    for i in xrange(0,N):
        d3=a2[i,:]-labels[i,:]
        d3=np.reshape(d3,(d3.shape[0],1))
        
        gradW2+=(d3*a1[i,:]).T
        gradb2+=d3.T
        
        d2=W2.dot(d3)
        g2=sigmoid_grad(a1[i,:])
        g2=np.reshape(g2,(g2.shape[0],1))
        d2=d2*g2
        gradW1+=(d2*data[i,:]).T
        gradb1+=d2.T
        
    gradW1 = gradW1/N
    gradb1 = gradb1/N
    gradW2 = gradW2/N
    gradb2 = gradb2/N
   
    
    ### Stack gradients 
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

#----------------------------------------------

# Check your sigmoid implementation
x=np.array([[1,2],[-1,-2]])
f=sigmoid(x)
g=sigmoid_grad(f)
print f
print g

# Sanity check for the gradient checker
quad = lambda x:(np.sum(x**2),x*2)

gradcheck_naive(quad,np.array(123.456))
gradcheck_naive(quad,np.random.randn(3,))
gradcheck_naive(quad,np.random.randn(4,5))

# Perform gradcheck on your neural network
gradcheck_naive(lambda params:forward_backward_prop(data,labels,params),params)
