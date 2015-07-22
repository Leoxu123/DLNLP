#!/usr/env/bin python

'''
    in this part,you're goint to implement
    1)A sigmoid activation function and its gradient
    2)A forward propagation for a simple neural network with cross-entropy cost
    3)A backward propagation algorithm to compute gradients for the parameters
    4)Gradient / derivative check
'''

import numpy as np
import random
def sigmoid(x):
    """Sigmoid function"""
    x=1/(1+np.exp(-x))
    return x

def sigmoid_grad(f):
    """Sigmoid gradient function"""
    f=f*(1-f)
    return f

# Check your sigmoid implementation
x=np.array([[1,2],[-1,-2]])
f=sigmoid(x)
g=sigmoid_grad(f)
print f
print g

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

# Sanity check for the gradient checker

quad = lambda x:(np.sum(x**2),x*2)

gradcheck_naive(quad,np.array(123.456))
gradcheck_naive(quad,np.random.randn(3,))
gradcheck_naive(quad,np.random.randn(4,5))

