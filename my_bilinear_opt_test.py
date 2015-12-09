import tensorflow as tf
import numpy as np

H = 10
M = 100
N = 1000
ALPHA = 1e0

def makeVariable (shape, stddev, wd, name, useL1Norm = False):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
	if useL1Norm:
		weight_decay = tf.mul(tf.reduce_sum(tf.abs(var)), wd, name='weight_loss_l1')
	else:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss_l2')
        tf.add_to_collection('losses', weight_decay)
        return var

def calcRMSE (y, yhat):
	return (y - yhat).T.dot(y - yhat)

def mySolveLS (A, b):
	return np.linalg.solve(A.T.dot(A) + np.eye(A.shape[1])*ALPHA, A.T.dot(b))

def fixGradients (grads_and_vars):
	MAX = 1e2
	return [ (tf.maximum(tf.minimum(gv[0], MAX), -MAX), gv[1]) for gv in grads_and_vars ]

def optTF (X, y):
	y = np.atleast_2d(y).T
	with tf.Graph().as_default():
		session = tf.InteractiveSession()
		x = tf.placeholder("float", shape=[None, X.shape[1]])
		y_ = tf.placeholder("float", shape=[None, 1])

		L = makeVariable([X.shape[1], H], stddev=1, wd=ALPHA, name="L")
		p = makeVariable([H, 1], stddev=1, wd=ALPHA, name="p", useL1Norm=False)

		#yhat = tf.matmul(tf.matmul(x, L), p)
		yhat = tf.matmul(tf.nn.relu(tf.matmul(x, L)), p)

		loss = tf.reduce_sum(tf.square(y_ - yhat), name='loss')
		tf.add_to_collection('losses', loss)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		#train_step = tf.train.GradientDescentOptimizer(learning_rate=.0001).minimize(total_loss)
		#train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.2)
		train_step = tf.train.AdamOptimizer(learning_rate=.001)
		grads_and_vars = train_step.compute_gradients(total_loss)
		grads_and_vars = fixGradients(grads_and_vars)
		op = train_step.apply_gradients(grads_and_vars)

		session.run(tf.initialize_all_variables())
		BATCH_SIZE = 100
		NUM_EPOCHS = 10000
		for i in range(NUM_EPOCHS):
			offset = i*BATCH_SIZE % (X.shape[0] - BATCH_SIZE)
						#op.run({x: X[offset:offset+BATCH_SIZE, :], y_: y[offset:offset+BATCH_SIZE, :]})
			#train_step.run({x: X[offset:offset+BATCH_SIZE, :], y_: y[offset:offset+BATCH_SIZE, :]})
			op.run({x: X[offset:offset+BATCH_SIZE, :], y_: y[offset:offset+BATCH_SIZE, :]})
			if i % 100 == 0:
				se = total_loss.eval({x: X, y_: y})
				print se
		return L.eval(), p.eval()

def optLS (X, y):
	yhat = np.zeros_like(y)
	lastRMSE = float('inf')
	TOLERANCE = 1e-3
	L = np.random.random((M, H))
	p = np.random.random(H)
	RMSE = calcRMSE(y, yhat)
	while np.abs(lastRMSE - RMSE) > TOLERANCE:
		# Step 1
		p = mySolveLS(X.dot(L), y)

		# Step 2
		Lvec = mySolveLS(np.kron(p, X), y)
		L = Lvec.reshape((L.shape[1], L.shape[0])).T

		yhat = X.dot(L).dot(p)
		lastRMSE = RMSE
		RMSE = calcRMSE(y, yhat)
		print RMSE
	return L, p

X = np.random.random((N, M))
y = np.random.random(N)
print "LS"
L1, p1 = optLS(X, y)
print "TF"
L2, p2 = optTF(X, y)
