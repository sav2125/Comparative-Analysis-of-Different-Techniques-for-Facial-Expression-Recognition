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

gc.collect()
print ("before open")
dataset = open('fer2013.csv', 'rb')
print ("after open")
dataset.readline()
print ("after readline")
reader = csv.reader(dataset, delimiter=',')
Y = []
testY = []
X = []
testX = []
count=1
count1=1
for row in reader:
	count+=1
	if row[2] == 'Training':
		emotion = row[0]
		Y.append(emotion)
		pixels = row[1].split(' ')
		X.append(pixels)
	else:
             emotion = row[0]
             testY.append(emotion)
             pixels = row[1].split(' ')
             testX.append(pixels)
print ("after loop")
Y = np.asfortranarray(Y, dtype=np.float)
Y = np.asfortranarray(Y.reshape(Y.shape[0],1))
X = np.asfortranarray(X, dtype=np.float)
X = spams.normalize(X)
testY = np.asfortranarray(testY, dtype=np.float)
testY = np.asfortranarray(testY.reshape(testY.shape[0],1))
testX = np.asfortranarray(testX, dtype=np.float)
W0=np.zeros((X.shape[1],Y.shape[1]),order="FORTRAN")
k = 2
"""
a = np.arange(36)
print a
a2d = np.reshape(a, (6,6) )
a = np.copy( np.reshape(blockshaped(a2d, k, k),(1,36)) )
print a
"""
print(X.shape)
print(Y.shape)
W = [i for i in xrange(7)]
for output_class in xrange(7):
    YY = (Y==output_class).astype(np.float).reshape(Y.shape)
    print YY.shape, Y.shape
    for i in xrange(X.shape[0]):
        X2D = np.reshape(X[i], (48,48) )
        X[i] = np.copy( np.reshape(blockshaped(X2D, k, k),(1,2304)) )
    
    param = {'numThreads' : -1,'verbose' : True,
             'lambda1' : 0.1, 'it0' : 10, 'max_it' : 1000,
             'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
             'pos' : False}
    param['regul']='group-lasso-l2'
    param['size_group']= k*k
    param['loss']='logistic'
    (W[output_class],optim_info)=spams.fistaFlat(YY,X,W0,True,**param)

count=1
for i in xrange(testX.shape[0]):
	prob=[]
	for k in xrange(7):
		prob.append(1/(1+np.exp(np.dot(-X,W[k]))))
	prob=np.asarray(prob)
	if(np.argmax(prob)==testY[i]):
		count=count+1    
print("test accuracy=",(count*1.0/testX.shape[0])*100)
print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:],0),np.mean(optim_info[3,:],0))


'''
print ("after read")
print(type(X))
print (Y.shape)
print (X.shape)
print (testY.shape)
print (testX.shape)
feature = X.shape[1]
print (feature)
print("d type")
D = []
print(type(D))
col = (int)(math.sqrt(feature))
append_row = np.zeros((1,feature))
'''

