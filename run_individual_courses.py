import tensorflow as tf
import common
import util
import numpy as np
import pandas
import sklearn.metrics

NUM_EPOCHS = 1000
BATCH_SIZE = 1000

def runNN (train_x, train_y, test_x, test_y, numHidden):
	print "NN({})".format(numHidden)
	session = tf.InteractiveSession()

	x = tf.placeholder("float", shape=[None, train_x.shape[1]])
	y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])
	keep_prob = tf.placeholder("float")

	W1 = tf.Variable(tf.truncated_normal([train_x.shape[1],numHidden], stddev=0.01))
	b1 = tf.Variable(tf.truncated_normal([numHidden], stddev=0.01))
	W2 = tf.Variable(tf.truncated_normal([numHidden,train_y.shape[1]], stddev=0.01))
	b2 = tf.Variable(tf.truncated_normal([train_y.shape[1]], stddev=0.01))

	level1 = tf.nn.relu(tf.matmul(x,W1) + b1)
	level2 = tf.nn.dropout(level1, keep_prob)
	level3 = tf.nn.softmax(tf.matmul(level2,W2) + b2)
	y = level3

	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.1).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(cross_entropy)

	session.run(tf.initialize_all_variables())
	for i in range(NUM_EPOCHS):
		offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
		train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: train_y[offset:offset+BATCH_SIZE, :], keep_prob: 0.1})
		if i % 100 == 0:
			util.showProgress(cross_entropy, x, y, y_, test_x, test_y, keep_prob)
	session.close()

def loadDataset (filename):
	d = pandas.io.parsers.read_csv(filename)
	d = util.getNonNullData(d)
	return pandas.get_dummies(d, columns = [ 'continent', 'LoE', 'gender' ])

if __name__ == "__main__":
	trainingSet = loadDataset("train_individual_courses.csv")
	testingSet = loadDataset("test_individual_courses.csv")

	# For one course (MCB63X), arbitrarily chosen from those that started after
	# the arbitrarily chosen cutoff date T (2015-06-30), predict certification conditional
	# on participation. Note that the first column for each course in the datasets indicates
	# whether the student started the course *before* T; the second column (hence "+ 1")
	# indicates whether the student started the course *after* T; and the third column
	# is whether he/she certified.
	COURSE = "MCB63X"
	columns = trainingSet.columns
	startCourseBeforeTIdx = np.argmax([ COURSE in col for col in columns ]) + 0
	startCourseAfterTIdx = startCourseBeforeTIdx + 1
	certifiedIdx = startCourseBeforeTIdx + 2
	# As features, we may use all columns corresponding to whether the student started a course
	# *before* time T (since we will have access to those data) *along with* demographic data.
	demographicIdxs = set([ i for i in range(len(columns)) if "LoE" in columns[i] ]).union(\
	  [ i for i in range(len(columns)) if "continent" in columns[i] ]).union(\
	  [ i for i in range(len(columns)) if "gender" in columns[i] ]).union(\
	  [ i for i in range(len(columns)) if "YoB" in columns[i] ])
	featureIdxs = list(set([ i for i in range(len(columns)) if "before" in columns[i] ]).union(demographicIdxs) - set([ startCourseBeforeTIdx ]))
	#featureIdxs = list(set([ i for i in range(len(columns)) if "before" in columns[i] ]) - set([ startCourseBeforeTIdx ]))
	#featureIdxs = list(demographicIdxs)
	trainingSet = trainingSet.as_matrix()
	testingSet = testingSet.as_matrix()

	# Only analyze people who participated in COURSE starting *after* T.
	idxs = np.nonzero(trainingSet[:,startCourseAfterTIdx] == 1)[0]
	trainingSet = trainingSet[idxs,:]
	idxs = np.nonzero(testingSet[:,startCourseAfterTIdx] == 1)[0]
	testingSet = testingSet[idxs,:]

	# Target variable is whether students who participated also *certified*
	train_x = trainingSet[:,featureIdxs]
	train_y = util.makeLabels(trainingSet[:,certifiedIdx])
	test_x = testingSet[:,featureIdxs]
	test_y = util.makeLabels(testingSet[:,certifiedIdx])

	# Scale data
	mx = np.mean(train_x, axis=0)
	sx = np.std(train_x, axis=0)
	sx[sx == 0] = 1
	train_x -= np.tile(np.atleast_2d(mx), (train_x.shape[0], 1))
	train_x /= np.tile(np.atleast_2d(sx), (train_x.shape[0], 1))
	# Scale testing data using parameters estimated on training set
	test_x -= np.tile(np.atleast_2d(mx), (test_x.shape[0], 1))
	test_x /= np.tile(np.atleast_2d(sx), (test_x.shape[0], 1))

	for numHidden in range(2, 20, 2):
		runNN(train_x, train_y, test_x, test_y, numHidden)
