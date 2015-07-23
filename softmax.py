#!/usr/env/bin python
# -*- coding: utf-8 -*-

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

# test the function
if __name__=="__main__":
    print softmax(np.array([[1001,1002],[3,4]]))
    print softmax(np.array([[-1001,-1002]]))

