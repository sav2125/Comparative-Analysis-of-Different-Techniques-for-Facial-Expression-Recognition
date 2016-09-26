import numpy as np
import math

import csv
import random
#from l1 import l1
from cvxpy import *
import scipy as scipy
import cvxopt as cvxopt
from cvxopt import *

def fusedLasso(trainExample = 30,testExample = 3,lambda1=0.05,lambda2=0.05):
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
     print cnt
     
     X = np.asfortranarray(X, dtype=np.int32)
     print("X shape : ")
     print(X.shape)
     Y = X[:,-1]
     print("Y shape : ")
     print(Y.shape)
     X = X[:,:2304]
     print("X shape : ")
     print(X.shape)
     print("trainExample : ")
     print(trainExample)
     Y = Y.reshape((trainExample,1))
     print("Y shape : ")
     print(Y.shape)
     count = 0

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
     print cnt
     testX = np.asfortranarray(testX, dtype=np.int32)
     print("testX shape : ")
     print(testX.shape)
     testY = testX[:,-1]
     print("testY shape : ")
     print(testY.shape)
     testX = testX[:,:2304]
     print("testX shape : ")
     print(testX.shape)
     print("testExample : ")
     print(testExample)
     testY = testY.reshape((testExample,1))
     print("testY shape : ")
     print(testY.shape)
     count = 0

     print ("after read")
     print(type(X))
     print (Y.shape)
     print (X.shape)
     print (testY.shape)
     print (testX.shape)
     feature = X.shape[1]
	#feature = 16
     print("d type")
     D = []
     print(type(D))
     col = (int)(math.sqrt(feature))
     append_row = np.zeros((1,feature))

     for i in xrange(0,feature):         
         r = i/col
         c = i%col
         if (r-1>=0 and c-1 >=0):
             
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r-1) + (c-1)] = -1
             append_row_copy[0, i] = 1
             D.append(append_row_copy)
         if (r-1>=0):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r-1) + c] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (r-1>=0 and c+1 < col ):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r-1) + (c+1)] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (c-1 >=0):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r) + (c-1)] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (c+1 < col ):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r) + (c+1)] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (r+1 < col and c-1 >=0):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r+1) + (c-1)] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (r + 1 < col ):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r+1) + c] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)
         if (r+1<  col and c+1 < col):
             append_row_copy = np.copy(append_row)
             append_row_copy[0,col*(r+1) + (c+1)] = -1
             append_row_copy[0, i] = 1		
             D.append(append_row_copy)	

     print("shape of D : ")
     print(len(D))
     D = np.asarray(D)
     D=np.reshape(D,(D.shape[0],D.shape[2]))
     print("type of D",type(D))
     print("shape of D",D.shape)

     m = X.shape[0]
     print m
     lambda1 = Parameter(sign="positive", value=0.05)
     lambda2 = Parameter(sign="positive", value=0.05)
     all_beta=[]
     for k in range(0,7):
         beta = Variable(feature,1)
#	a = Variable(feature)
#	b = Variable(len(D) , 1)
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
         objective = Minimize( sum_entries(incorX*beta) + sum_entries(logistic(-X*beta)) + lambda1*norm(beta,1) +lambda2*norm(D*beta,1) )
#		constraints = [a == beta , b == D*beta]
         prob = Problem(objective)#,constraints)
         prob.solve(solver=SCS)
         np.save('beta'+str(k)+'.npy',beta.value)
         all_beta.append(np.asarray(beta.value))
         print("Iteration",k)

#print testX[3]
#print all_beta[1]
#print np.dot(-testX[3],all_beta[1])
     count=0
     for i in xrange(testX.shape[0]):
         prob = []
         for k in xrange(7):
             prob.append(1/(1+np.exp(np.dot(-testX[i],all_beta[k]))))
         prob=np.asarray(prob)
         if(np.argmax(prob)==testY[i][0]):
             count=count+1

     print("test accuracy=",(count*1.0/testX.shape[0])*100)
     print(count)
     print(testX.shape[0])
	# Check for error.
	#if prob.status != OPTIMAL:  #	 raise Exception("Solver did not converge!")
