import tensorflow as tf
import common
import util
import numpy as np
import pandas
import sklearn.metrics

NUM_EPOCHS = 2000
BATCH_SIZE = 100

def makeVariable (shape, stddev, wd):
	var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
	tf.add_to_collection('losses', weight_decay)
	return var

def runMLR (train_x, train_y, test_x, test_y, numEpochs = NUM_EPOCHS):
	print "MLR"
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x = tf.placeholder("float", shape=[None, train_x.shape[1]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		W1 = makeVariable([train_x.shape[1],numHidden], stddev=0.5, wd=1e0)
		b1 = makeVariable([train_y.shape[1]], stddev=0.5, wd=1e0)

		y = tf.nn.softmax(tf.matmul(x,W1) + b1)

		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		train_step = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: train_y[offset:offset+BATCH_SIZE, :]})
			if i % 100 == 0:
				util.showProgress(cross_entropy, x, y, y_, test_x, test_y)
		session.close()

def runNN (train_x, train_y, test_x, test_y, numHidden, numEpochs = NUM_EPOCHS):
	print "NN({})".format(numHidden)
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x = tf.placeholder("float", shape=[None, train_x.shape[1]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])
		keep_prob = tf.placeholder("float")

		W1 = makeVariable([train_x.shape[1],numHidden], stddev=0.05, wd=1e1)
		b1 = makeVariable([numHidden], stddev=0.05, wd=1e1)
		W2 = makeVariable([numHidden,train_y.shape[1]], stddev=0.05, wd=1e0)
		b2 = makeVariable([train_y.shape[1]], stddev=0.05, wd=1e0)

		level1 = tf.nn.relu(tf.matmul(x,W1) + b1)
		level2 = tf.nn.dropout(level1, keep_prob)
		level3 = tf.nn.softmax(tf.matmul(level2,W2) + b2)
		y = level3

		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.1).minimize(total_loss)
		#train_step = tf.train.AdamOptimizer(learning_rate=.001).minimize(total_loss)

		session.run(tf.initialize_all_variables())
		for i in range(numEpochs):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: train_y[offset:offset+BATCH_SIZE, :], keep_prob: 0.1})
			if i % 100 == 0:
				util.showProgress(cross_entropy, x, y, y_, test_x, test_y, keep_prob)
		session.close()

def loadDataset (filename, courseId, T, requiredCols = None):
	d = pandas.io.parsers.read_csv(filename)
	d.start_time = d.start_time.astype(np.datetime64)

	# Only analyze rows belonging to users who participated in the courseId after T
	afterIdxs = np.nonzero(d.start_time >= T)[0]
	e = d.iloc[afterIdxs]
	idxs = np.nonzero(e.course_id == courseId)[0]
	userIds = set(e.iloc[idxs].user_id)
	d = d.iloc[np.nonzero(d.user_id.isin(userIds))[0]]

	# Compute labels (i.e., whether each user explored the specified course)
	userIdsLabelsMap = dict(zip(e.user_id, e.explored))
		
	# Only analyze rows occurring before T
	idxs = np.nonzero(d.start_time < T)[0]
	d = d.iloc[idxs]
	
	# Assemble demographics matrix
	demographics = d.drop_duplicates("user_id")
	y = demographics.user_id.map(userIdsLabelsMap).as_matrix()
	demographics = demographics[common.DEMOGRAPHIC_FIELDS + [ "user_id" ]]
	demographics = pandas.get_dummies(demographics, columns=["continent", "LoE", "gender"])
	demographics = demographics.sort("user_id")
	
	# Assemble prior course matrix
	nDaysAct = d.ndays_act
	courses = pandas.get_dummies(d[['course_id', 'user_id']], columns=["course_id"])
	courseCols = [ col for col in courses.columns if "course_id" in col ]
	courses[courseCols] *= np.tile(np.atleast_2d(nDaysAct).T, (1, len(courseCols)))  # Multiply course indicator vars by ndays_act
	courses = courses.sort("user_id")
	# Aggregate within each user
	courses = courses.groupby(['user_id']).sum()

	# Combine demographics and course matrices into x
	courses = courses.reset_index()
	demographics = demographics.reset_index()
	#x = demographics  # DEMOGRAPHICS ONLY
	x = pandas.concat([ demographics, courses ], axis=1)  # ALL FEATURES
	x = x.drop('index', 1)
	x = x.drop('user_id', 1)
	if requiredCols != None:
		for colName in x.columns:
			if colName not in requiredCols:
				x = x.drop(colName, 1)
		for colName in requiredCols:
			if colName not in x.columns:
				x[colName] = pandas.Series(np.zeros(x.shape[0]), index=x.index)
	x = x.reindex_axis(sorted(x.columns), axis=1)  # Sort by column names
	return x.as_matrix(), util.makeLabels(y), x.columns

def computeT (courseId):
	# Find the 1st percentile, over all .csv files, of the start_time of the specified course
	startTimes = []
	for filenameRoot in [ "train", "test", "holdout" ]:
		filename = "{}_individual.csv".format(filenameRoot)
		d = pandas.io.parsers.read_csv(filename)
		idxs = np.nonzero(d.course_id == courseId)[0]
		d = d.iloc[idxs]
		startTimes += list(d.start_time.astype(np.datetime64))
	START_T = np.sort(startTimes)[int(0.01 * len(startTimes))]  # Use 1st percentile als proxy for course start date
	T = START_T + np.timedelta64(28, 'D')  # Add 4 weeks
	return T

def initializeAllData (courseId):
	T = computeT(courseId)
	train_x, train_y, colNames = loadDataset("train_individual.csv", courseId, T)
	test_x, test_y, _ = loadDataset("test_individual.csv", courseId, T, list(colNames))

	# Normalize data
	normalize = True
	if normalize:
		mx = np.mean(train_x, axis=0)
		sx = np.std(train_x, axis=0)
		sx[sx == 0] = 1
		train_x -= np.tile(np.atleast_2d(mx), (train_x.shape[0], 1))
		train_x /= np.tile(np.atleast_2d(sx), (train_x.shape[0], 1))
		# Scale testing data using parameters estimated on training set
		test_x -= np.tile(np.atleast_2d(mx), (test_x.shape[0], 1))
		test_x /= np.tile(np.atleast_2d(sx), (test_x.shape[0], 1))

	return train_x, train_y, test_x, test_y, colNames

def runLLL_NN (all_train_x, all_train_y, all_test_x, all_test_y):
	pass

# Life-long learning (LLL) experiment
def runLLLExperiments ():
	# Converts from a two-column binary label vector into a matrix with nCols
	# and copies y into the matrix at the specified index.
	def convertY (y, index, nCols):
		convertedY = np.zeros((y.shape[0], nCols), dtype=y.dtype)
		convertedY[:, 2*index] = y[:, 0]
		convertedY[:, 2*index+1] = y[:, 1]
		return convertedY

	# Get list of all course_id's
	d = pandas.io.parsers.read_csv("train_individual.csv")
	courseIds = np.unique(d.course_id)

	# Initialize training and testing matrices
	train_x, train_y, colNames = loadDataset("train_individual.csv", courseIds[0], computeT(courseIds[0]))
	all_train_x = np.zeros((0, train_x.shape[1]), dtype=train_x.dtype)
	all_train_y = np.zeros((0, train_y.shape[1] * len(courseIds)), dtype=train_y.dtype)
	all_test_x = np.zeros((0, train_x.shape[1]), dtype=train_x.dtype)
	all_test_y = np.zeros((0, train_y.shape[1] * len(courseIds)), dtype=train_y.dtype)
	
	# Collect training and testing data for all courses
	print "Loading data for..."
	for i, courseId in enumerate(courseIds):
		print courseId
		T = computeT(courseId)
		train_x, train_y, _ = loadDataset("train_individual.csv", courseId, T, list(colNames))
		test_x, test_y, _ = loadDataset("test_individual.csv", courseId, T, list(colNames))

		all_train_x = np.vstack((all_train_x, train_x))
		all_train_y = np.vstack((all_train_y, convertY(train_y, i, len(courseIds) * 2)))
		all_test_x = np.vstack((all_test_x, test_x))
		all_test_y = np.vstack((all_test_y, convertY(test_y, i, len(courseIds) * 2)))

	# Normalize data
	mx = np.mean(all_train_x, axis=0)
	sx = np.std(all_train_x, axis=0)
	sx[sx == 0] = 1
	all_train_x -= np.tile(np.atleast_2d(mx), (all_train_x.shape[0], 1))
	all_train_x /= np.tile(np.atleast_2d(sx), (all_train_x.shape[0], 1))
	# Scale testing data using parameters estimated on training set
	all_test_x -= np.tile(np.atleast_2d(mx), (all_test_x.shape[0], 1))
	all_test_x /= np.tile(np.atleast_2d(sx), (all_test_x.shape[0], 1))

	runLLL_NN(all_train_x, all_train_y, all_test_x, all_test_y)

def runNNExperiments ():
	#COURSE_ID = "HarvardX/SW12.5x/2T2014"
	COURSE_ID = "HarvardX/ER22.1x/1T2014"
	if 'train_x' not in globals():
		train_x, train_y, test_x, test_y, colNames = initializeAllData(COURSE_ID)

	for numHidden in range(2, 20, 2):
		runNN(train_x, train_y, test_x, test_y, numHidden)

if __name__ == "__main__":
	runLLLExperiments()
	#runNNExperiments()
