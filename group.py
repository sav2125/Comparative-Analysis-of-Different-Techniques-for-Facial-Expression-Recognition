# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:12:08 2016

@author: Siddharth Aman Varshney
"""

import numpy as np
import math
import csv
from cvxpy import *
import scipy as scipy
import gc
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def groupLasso(trainExample,testExample,groupSize,lambda1):
    gc.collect()
    print ("before open")
    dataset = open('dataset_train_plus_publictest.csv', 'rb')
    print ("after open")
    dataset.readline()
    print ("after readline")
    reader = csv.reader(dataset, delimiter=',')
    count = 0
    X = []
    testX = []
    randtrain = random.sample(range(32298),trainExample)
    randtest = random.sample(range(3589),testExample)
    count = 0
    cnt = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    print("count is : ")
    for row in reader:
        count = count + 1
        if(cnt > trainExample):
            break
        if(count in randtrain):
            X.append(row)
            cnt = cnt + 1
        #print cnt
    
    #Y = np.asfortranarray(Y, dtype=np.int32)
    #Y = np.asfortranarray(Y.reshape(Y.shape[0],1))
    X = np.asfortranarray(X, dtype=np.int32)
    Y = X[:,-1]
    X = X[:,:2304]
    Y = Y.reshape((trainExample,1))
    count = 0

    #testY = np.asfortranarray(testY, dtype=np.int32)
    #testY = np.asfortranarray(testY.reshape(testY.shape[0],1))
    #testX = np.asfortranarray(testX, dtype=np.int32)
    testdataset = open('dataset_privatetest.csv', 'rb')
    print ("after open")
    testdataset.readline()
    print ("after readline")
    reader2 = csv.reader(testdataset, delimiter=',')
    count = 0
    cnt = 0
    print("count is : ")
    for row2 in reader2:
        count = count + 1
        if(cnt > testExample):
            break
        if(count in randtest):
            testX.append(row2)
            cnt = cnt + 1
        #print cnt
    testX = np.asfortranarray(testX, dtype=np.int32)
    testY= testX[:,-1]
    count = 0
    testX = testX[:,:2304]
    testY = testY.reshape((testExample,1))
    
    print ("after read")
    print(type(X))
    print (Y.shape)
    print (X.shape)
    print (testY.shape)
    print (testX.shape)
    W0=np.zeros((X.shape[1],Y.shape[1]),order="FORTRAN")
    k = groupSize
    """
    a = np.arange(36)
    print a
    a2d = np.reshape(a, (6,6) )
    a = np.copy( np.reshape(blockshaped(a2d, k, k),(1,36)) )
    print a


    """

    W = [i for i in xrange(7)]
    for output_class in xrange(7):
        YY = (Y==output_class).astype(np.float).reshape(Y.shape)
        print YY.shape, Y.shape
        for i in xrange(X.shape[0]):
            X2D = np.reshape(X[i], (48,48) )
            X[i] = np.copy( np.reshape(blockshaped(X2D, k, k),(1,2304)) )
            
    feature = X.shape[1]
    multiply_factor = (k*k)
    Dsize = X.shape[1]/(multiply_factor)
    D = np.zeros((Dsize,feature))
    for row_index in range(Dsize):
        offset = multiply_factor * row_index
        for column_index in xrange(multiply_factor):
            D[row_index][offset + column_index] = 1
    print('D.shape',D.shape)
    m = X.shape[0]
    
    all_beta=[]
    for k in range(0,7):
        beta = Variable(feature, 1 )
        a = Variable(feature)
        b = Variable(len(D) , 1)
        #b = Variable(len(D), D.shape[1])
        corY = []
        corX = []
        incorX = []
        incorY = []
        for i in range(m):
            if Y[i] == k:
                corY.append(Y[i])
                corX.append(X[i])
            else:
			incorX.append(X[i])
			incorY.append(Y[i])
        corY = np.asarray(corY)
        corX = np.asarray(corX)
        incorX = np.asarray(incorX)
        incorY = np.asarray(incorY)
        print corX.shape
        objective = Minimize( sum_entries(incorX*beta) + sum_entries(logistic(-X*beta))  +lambda1*norm(D*(square(beta)),1) )
        #	constraints = [b == D*square(beta)]
        prob = Problem(objective)#,constraints)
        prob.solve(solver=SCS)
        np.save('groupBeta'+str(k)+'.npy',beta.value)    
        all_beta.append(np.asarray(beta.value))
      
        print "Iteration",k

    count=0
    for i in xrange(testX.shape[0]):
        prob=[]
        for k in xrange(7):
            prob.append(1/(1+np.exp(np.dot(-testX[i],all_beta[k]))))
            prob=np.asarray(prob)
        if(np.argmax(prob)==testY[i][0]):
                count=count+1
    print("test accuracy=",(count*1.0/testX.shape[0])*100)
