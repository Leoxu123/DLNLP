import random
import numpy as np

def softmax(x):
    """Softmax function"""

    """for softmax ,we have softmax(x)=softmax(x+c)
        using this property to deal with large values,
        such as exp(1000) will overflow
    """
    row_min = np.min(x,axis=1)
    row_min = np.reshape(row_min,(row_min.shape[0],1))
    x=x-row_min
    y=np.sum(np.exp(x),axis=1)
    y=np.reshape(y,(y.shape[0],1))
    x=np.exp(x)/y

    return x

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

def normalizeRows(x):
    """ Row normalization function """
    
    ### YOUR CODE HERE
    x2 = x*x
    ### END YOUR CODE
    if len(np.shape(x)) <= 1:
        return np.sqrt(x2 / np.sum(x2))
    
    return np.sqrt(x2/(np.sum(x2, -1)[:, np.newaxis]))

if __name__=="__main__":
    print 'softmax test:'
    print softmax(np.array([[1001,1002],[3,4]]))
    print softmax(np.array([[-1001,-1002]]))

    print 'sigmoid and its gradients test:'
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    print g

    print 'normalize function test:'
    print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
