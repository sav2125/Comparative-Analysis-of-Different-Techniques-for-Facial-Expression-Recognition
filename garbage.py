"""
beta=[]
b = X[1].shape[0]
print b
A = None
temp=matrix(X)
x=None
def F(x=None, z=None,A=None):
   if x is None: return 0, matrix(0.0, (feature,1))
   print type(temp)
   w = exp(temp*x)
   lamda=0.1
   B = np.array(A)
   l1_norm  =  np.linalg.norm(B,axis  = 1)
   f = c.T*x + sum(log(1+w)) + l1_norm
   grad = c + temp.T * div(w, 1+w)
   if z is None: return f, grad.T
   H = temp.T * spdiag(div(w,(1+w)**2)) * temp
   return f, grad.T, z[0]*H
"""
"""
for k in range(0,7):
	c = matrix(0.0 , (feature,1) )
	#print c
	for i in xrange(0,m):
		if (Y[i] == k):
			for j in xrange(0,feature):
				c[j,0] = c[j,0] + X[i,j]
	sol = solvers.cp(F,A=x)
	beta.append(sol['x'])
print beta
"""