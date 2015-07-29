#!/usr/env/bin python
# -*- coding: utf-8 -*-

import numpy as np
import random
from cs224d.data_utils import *
from utils import *

dataset = type('dummy',(),{})()

def dummySampleTokenIdx():
    return random.randint(0,4)

def getRandomContext(C):
    tokens = ['a','b','c','d','e']
    return tokens[random.randint(0,4)],[tokens[random.randint(0,4)] for i in xrange(2*C)]

dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       # 
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    costArr=np.sum(outputVectors*predicted,axis=1)
    costArr= softmax(np.array([costArr]))
  
    cost = -np.log(costArr[0,target])
    gradPred = np.sum(outputVectors.T*costArr,axis=1)- outputVectors[target,:]
    
    grad = np.outer(costArr,predicted)
    grad[target,:]-=predicted
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    sampleidx=[dataset.sampleTokenIdx() for i in range(0,K)]
    costPositive = np.log(sigmoid(np.inner(predicted,outputVectors[target,:])))
    costNegative = np.sum([np.log(1-sigmoid(np.inner(predicted,outputVectors[idx,:]))) for idx in sampleidx ])
    cost = -costPositive - costNegative
    
    gradPred = (sigmoid(np.inner(predicted,outputVectors[target,:]))-1)*outputVectors[target,:]
    for idx in sampleidx:
        gradPred += (sigmoid(np.inner(outputVectors[idx,:],predicted)))*outputVectors[idx,:]
    grad = np.zeros(outputVectors.shape)
    grad[target,:] =  (sigmoid(np.inner(predicted,outputVectors[target,:]))-1)*predicted
    for idx in sampleidx:
        # do not use = here ,because idx may be the same here,thus we need update twice
        grad[idx,:] += (sigmoid(np.inner(predicted,outputVectors[idx,:])))*(predicted)
    
    ### END YOUR CODE
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #         
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    inidx = tokens[currentWord]
    predicted = inputVectors[inidx,:]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        target = tokens[word]
        cost1,gradPred,grad = word2vecCostAndGradient(predicted,target,outputVectors)
        cost += cost1
        gradIn[inidx,:] += gradPred
        gradOut+=grad
        
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #         
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    #inputVectors=inputVectors.T
    #outputVectors=outputVectors.T
    predicted = 0
    for word in contextWords:
        predicted+=inputVectors[tokens[word],:]
    cost = 0
    target = tokens[currentWord]
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    cost,gradPred,grad = word2vecCostAndGradient (predicted,target,outputVectors)
    for word in contextWords:
        gradIn[tokens[word],:] += gradPred
    gradOut = grad
    ### END YOUR CODE
    return cost, gradIn, gradOut
	
# Gradient check!

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad
