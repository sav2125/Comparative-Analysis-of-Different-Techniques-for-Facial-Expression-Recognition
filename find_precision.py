import numpy as np
import math
import csv
import ast
from l1 import l1
from cvxpy import *
import scipy as scipy
import cvxopt as cvxopt
from cvxopt import *

#def find_precision(fileName,testExample):
testX = []
testExample = 3000
randtest = random.sample(range(3589),testExample)
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
testY= testX[:,-1]
testX = testX[:,:2304]
testY = testY.reshape((testExample,1))
   
all_beta = []
fileName = 'groupBeta'
for i in xrange(7):
    beta = np.load(fileName+str(i)+'.npy')
    all_beta.append(beta)
count=0
for i in xrange(testX.shape[0]):
    prob=[]
    for k in xrange(7):
        prob.append(1/(1+np.exp(np.dot(-testX[i],all_beta[k]))))
    prob=np.asarray(prob)
    if(np.argmax(prob)==testY[i][0]):
        count=count+1
print "test accuracy=",(count*1.0/testX.shape[0])*100