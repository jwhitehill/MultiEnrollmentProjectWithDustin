import tensorflow as tf
import common
import util
import numpy as np
import pandas
import sklearn.metrics

NUM_EPOCHS = 1000
BATCH_SIZE = 1000

def makeLabels (y):
	labels = np.hstack((1 - np.atleast_2d(y).T, np.atleast_2d(y).T)).astype(np.float64)
	return labels

def showProgress (cross_entropy, x, y, y_, test_x, test_y):
	ll = cross_entropy.eval({x: test_x, y_: makeLabels(test_y)})
	auc = sklearn.metrics.roc_auc_score(test_y, y.eval({x: test_x})[:,1])
	print "LL={} AUC={}".format(ll, auc)

def runMLR (train_x, train_y, test_x, test_y):
	print "MLR"
	session = tf.InteractiveSession()

	x = tf.placeholder("float", shape=[None, train_x.shape[1]])
	y_ = tf.placeholder("float", shape=[None, 2])

	W = tf.Variable(tf.truncated_normal([train_x.shape[1],2], stddev=0.01))
	b = tf.Variable(tf.truncated_normal([2], stddev=0.01))
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.1).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(cross_entropy)

	session.run(tf.initialize_all_variables())
	for i in range(NUM_EPOCHS):
		offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
		train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: makeLabels(train_y[offset:offset+BATCH_SIZE])})
		if i % 100 == 0:
			showProgress(cross_entropy, x, y, y_, test_x, test_y)
	session.close()

def runNN (train_x, train_y, test_x, test_y, numHidden):
	print "NN({})".format(numHidden)
	session = tf.InteractiveSession()

	x = tf.placeholder("float", shape=[None, train_x.shape[1]])
	y_ = tf.placeholder("float", shape=[None, 2])

	W1 = tf.Variable(tf.truncated_normal([train_x.shape[1],numHidden], stddev=0.01))
	b1 = tf.Variable(tf.truncated_normal([numHidden], stddev=0.01))
	W2 = tf.Variable(tf.truncated_normal([numHidden,2], stddev=0.01))
	b2 = tf.Variable(tf.truncated_normal([2], stddev=0.01))

	z = tf.nn.relu(tf.matmul(x,W1) + b1)
	y = tf.nn.softmax(tf.matmul(z,W2) + b2)

	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.1).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(cross_entropy)

	session.run(tf.initialize_all_variables())
	for i in range(NUM_EPOCHS):
		offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
		train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: makeLabels(train_y[offset:offset+BATCH_SIZE])})
		if i % 100 == 0:
			showProgress(cross_entropy, x, y, y_, test_x, test_y)
	session.close()

if __name__ == "__main__":
	trainingSet = util.loadDataset("train.csv")
	testingSet = util.loadDataset("test.csv")
	fields = list(testingSet.columns)
	fields.remove(common.TARGET_VARIABLE)

	train_y = trainingSet[common.TARGET_VARIABLE].as_matrix() > 1
	train_x = trainingSet[fields].as_matrix().astype(np.float32)
	test_y = testingSet[common.TARGET_VARIABLE].as_matrix() > 1
	test_x = testingSet[fields].as_matrix().astype(np.float32)

	# Scale data
	mx = np.mean(train_x, axis=0)
	sx = np.std(train_x, axis=0)
	sx[sx == 0] = 1
	train_x -= np.tile(np.atleast_2d(mx), (train_x.shape[0], 1))
	train_x /= np.tile(np.atleast_2d(sx), (train_x.shape[0], 1))
	# Scale testing data using parameters estimated on training set
	test_x -= np.tile(np.atleast_2d(mx), (test_x.shape[0], 1))
	test_x /= np.tile(np.atleast_2d(sx), (test_x.shape[0], 1))

	runMLR(train_x, train_y, test_x, test_y)
	for numHidden in range(2, 20, 2):
		runNN(train_x, train_y, test_x, test_y, numHidden)
