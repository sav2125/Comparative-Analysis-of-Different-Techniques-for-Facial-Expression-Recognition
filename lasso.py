import numpy as np
import math
import csv
from l1 import l1
from cvxpy import *
import scipy as scipy
import cvxopt as cvxopt
from cvxopt import *
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
count = 1
for row in reader:
	count = count + 1
	if count ==500:
		break
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
Y = np.asfortranarray(Y, dtype=np.int32)
Y = np.asfortranarray(Y.reshape(Y.shape[0],1))
X = np.asfortranarray(X, dtype=np.int32)
testY = np.asfortranarray(testY, dtype=np.int32)
testY = np.asfortranarray(testY.reshape(testY.shape[0],1))
testX = np.asfortranarray(testX, dtype=np.int32)
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
D = np.asarray(D)#
D=np.reshape(D,(D.shape[0],D.shape[2]))
print("type of D",type(D))
print("shape of D",D.shape)

m = X.shape[0]
lambda1 = Parameter(sign="positive", value=0.1)
lambda2 = Parameter(sign="positive", value=0.1)
all_beta=[]
for k in range(0,2):
	beta = Variable(feature,1)
	a = Variable(feature)
	b = Variable(len(D) , 1)
	YTX = np.dot(Y.T,X)
	'''
	print ("be4 objective")
	print("Y shape",Y.T.shape)
	print("X shape",X.shape)
	#A1=np.dot(np.dot(Y.T,X)*beta)
	#print(A1.shape)
	#print(beta.shape)
	#objective = Minimize( sum_entries(X*beta) )
	#+  
	print ("after objective")
	'''
	objective = Minimize( YTX*beta - sum_entries(X*beta) + sum_entries(logistic(-X*beta)) + lambda1*norm(a,1)  )
	constraints = [a == beta ]
	prob = Problem(objective,constraints)
	prob.solve(solver=SCS)
	all_beta.append(beta)
	print("Iteration",k)

count=0
for i in xrange(testX.shape[0]):
	prob=[]
	for k in xrange(2):
		prob.append(1/(1+np.exp(np.dot(X,all_beta[k]))))
	prob=np.asarray(prob)
	if(np.argmax(prob)==testY[i]):
		count=count+1

print("test accuracy=",(count/testX.shape[0])*100)
	# Check for error.
	#if prob.status != OPTIMAL:
    #	 raise Exception("Solver did not converge!")
