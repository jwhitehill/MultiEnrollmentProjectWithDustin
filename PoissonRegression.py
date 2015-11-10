import numpy as np
import scipy.optimize

class PoissonRegression:
	"""
	Poisson regression
	"""
	def __init__ (self, C=1.0, fit_intercept = True, minValue = 0):
		self.C = C
		self.fit_intercept = fit_intercept
		self.minValue = minValue

	def getC (self):
		C = np.eye(self.theta.shape[0]) * self.C
		if self.fit_intercept:
			C[0][0] = 0  # No regularization on intercept term
		return C

	def fit (self, X, y, tol = 1e-5):
		X = np.array(X)
		if self.fit_intercept:
			X = np.hstack((X, np.ones((X.shape[0], 1))))
		y = np.array(y) - self.minValue
		self.theta = np.zeros(X.shape[1])

		# Use Newton's method
		gradient = lambda theta: - (y - self.predictHelper(X, theta)).dot(X) + self.getC().dot(theta)
		hessian = lambda theta: (X * np.tile(np.atleast_2d(self.predictHelper(X, theta)).T, X.shape[1])).T.dot(X) + self.getC()

		# Repeat until convergence
		while True:
			hess = hessian(self.theta)
			grad = gradient(self.theta)
			change = np.linalg.solve(hess, grad)
			theta = self.theta - change

			absdiff = np.sum(np.abs(self.predictHelper(X, theta) - self.predictHelper(X, self.theta)))
			if absdiff < tol:
				break
			print self.likelihood(X, y, theta)
			self.theta = theta
	
	def likelihood (self, X, y, theta):
		X = np.array(X)
		y = np.array(y)
		return y.dot(X.dot(theta)) - np.sum(self.predictHelper(X, theta)) - 0.5 * theta.dot(theta)*self.C
		
	def predict (self, X):
		if self.fit_intercept:
			X = np.hstack((X, np.ones((X.shape[0], 1))))
		return self.predictHelper(X, self.theta) + self.minValue

	def predictHelper (self, X, theta):
		return np.exp(X.dot(theta))

if __name__ == "__main__":
	X = [ [ 3, 2 ],
	      [ 2, 5 ],
	      [ -1, 4]
	    ]
	y = [ 1, 2, 3 ]
	pr = PoissonRegression(C=1e2)
	pr.fit(X, y)
