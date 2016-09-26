from cvxpy import *
import numpy

# Problem data.
m = 10
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m, 1)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_entries(square(A*x - b)))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

print "Optimal value", prob.solve()
print "Optimal var"
print x.value # A numpy matrix.