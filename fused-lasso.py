import numpy as np
import spams
import csv

dataset = open('fer2013.csv', 'rb')
dataset.readline()
reader = csv.reader(dataset, delimiter=',')
Y = []
testY = []
X = []
testX = []
for row in reader:
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
Y = np.asfortranarray(Y, dtype=np.float64)
Y = np.asfortranarray(Y.reshape(Y.shape[0],1))
X = np.asfortranarray(X, dtype=np.float64)
testY = np.asfortranarray(testY, dtype=np.float64)
testY = np.asfortranarray(testY.reshape(testY.shape[0],1))
testX = np.asfortranarray(testX, dtype=np.float64)

param = {'numThreads' : -1,'verbose' : True, 'lambda1' : 0.1, 'it0' : 10, 'max_it' : 300, 'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False, 'pos' : False}
param['loss'] = 'multi-logistic'
param['regul'] = 'fused-lasso'
param['lambda2'] = 0.1
param['lambda3'] = 0.0; #
nclasses = 7
W0 = np.zeros((X.shape[1],nclasses),dtype=np.float64,order="FORTRAN")
(W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)
print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(
optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))

print W.shape

testY_ = np.dot(testX, W)

correct = 0
for i in xrange(0,len(testY)):
#	print '\n'
#	print testY[i]
#	print testY_[i]
	testY_[i] = np.exp(testY_[i])
#	print testY_[i]
#	print testY[i]
#	print np.argmax(testY_[i])
#	print (testY[i] == np.argmax(testY_[i]))
	if testY[i] == np.argmax(testY_[i]):
		correct += 1

print correct*1.0/len(testY)