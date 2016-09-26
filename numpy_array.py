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
D = np.empty([1,feature])
print(type(D))
col = (int)(math.sqrt(feature))


for i in xrange(0,feature):
	print(i)
	r = i/col
	c = i%col
	if (r-1>=0 and c-1 >=0):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r-1) + (c-1)] = -1
		append_row[0, i] = 1
		D=np.vstack((D,append_row))
	if (r-1>=0):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r-1) + c] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (r-1>=0 and c+1 < col ):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r-1) + (c+1)] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (c-1 >=0):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r) + (c-1)] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (c+1 < col ):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r) + (c+1)] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (r+1 < col and c-1 >=0):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r+1) + (c-1)] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (r + 1 < col ):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r+1) + c] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))
	if (r+1<  col and c+1 < col):
		append_row = np.zeros((1,feature))
		append_row[0,col*(r+1) + (c+1)] = -1
		append_row[0, i] = 1		
		D=np.vstack((D,append_row))	

print("shape of D : ")
print(len(D))
D = np.asarray(D)
m = X.shape[0]
lambda1 = Parameter(sign="positive", value=0.1)
lambda2 = Parameter(sign="positive", value=0.1)
'''
for k in range(0,7):
	beta = Variable(feature)
	a = Variable(feature)
	b = Variable(len(D) , 1)
	print ("be4 objective")
	objective = Minimize( np.dot(np.dot(Y.T,X),beta) - sum_entries(np.dot(X,beta)) )
	print ("after objective")
	objective = Minimize( np.dot(np.dot(Y.T,X),beta) - sum_entries(np.dot(X,beta)) + sum_entries(np.log(1 + np.exp(np.dot(X,beta)))) + lambda1*norm(a,1)  )
	print ("after objective")
	constraints = [a == beta ]
	print ("after constraints")
	prob = Problem(objective)
	print ("after prob")
	prob.solve(solver=cvx.CVXOPT,verbose=True)
	print ("after solve")
	print beta
	temp1 = (Y.T)
'''