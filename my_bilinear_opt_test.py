import tensorflow as tf
import numpy as np

H = 4
M = 10
N = 1000

def calcRMSE (y, yhat):
	return (y - yhat).T.dot(y - yhat)

def mySolveLS (A, b, alpha):
	return np.linalg.solve(A.T.dot(A) + np.eye(A.shape[1]) * alpha*alpha, A.T.dot(b))

def optLS (X, y):
	yhat = np.zeros_like(y)
	lastRMSE = float('inf')
	TOLERANCE = 1e-4
	ALPHA = 1e1
	L = np.random.random((M, H))
	p = np.random.random(H)
	RMSE = calcRMSE(y, yhat)
	while np.abs(lastRMSE - RMSE) > TOLERANCE:
		# Step 1
		p = mySolveLS(X.dot(L), y, ALPHA)

		# Step 2
		Lvec = mySolveLS(np.kron(p, X), y, ALPHA)
		L = Lvec.reshape((L.shape[1], L.shape[0])).T

		yhat = X.dot(L).dot(p)
		lastRMSE = RMSE
		RMSE = calcRMSE(y, yhat)
		print RMSE

X = np.random.random((N, M))
y = np.random.random(N)
optLS(X, y)
