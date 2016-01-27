import tensorflow as tf
import util
import pandas
import math
import numpy as np
import sklearn.metrics
import sklearn.linear_model

BATCH_SIZE = 100

MIN_EXAMPLES = 10
START_DATES = {
	'HarvardX/SW25x/1T2014': np.datetime64('2014-02-25'),
	'HarvardX/SW12x/2013_SOND': np.datetime64('2013-10-31'),
	'HarvardX/SW12.2x/1T2014': np.datetime64('2014-01-02'),
	'HarvardX/SW12.3x/1T2014': np.datetime64('2014-02-13'),
	'HarvardX/SW12.4x/1T2014': np.datetime64('2014-03-20'),
	'HarvardX/SW12.5x/2T2014': np.datetime64('2014-04-24'),
	'HarvardX/SW12.6x/2T2014': np.datetime64('2014-05-22'),
	'HarvardX/SW12.7x/3T2014': np.datetime64('2014-09-04'),
	'HarvardX/SW12.8x/3T2014': np.datetime64('2014-10-09'),
	'HarvardX/SW12.9x/3T2014': np.datetime64('2014-11-20'),
	'HarvardX/SW12.10x/1T2015': np.datetime64('2015-01-08'),
	#'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27'),
	#'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27'),
	#'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27'),
	#'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27'),
	#'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27'),
	#'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.9x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20'),
}

PREDICTION_DATES = {
	'HarvardX/SW25x/1T2014': np.datetime64('2014-03-18 17:00:00'),
	'HarvardX/SW12x/2013_SOND': np.datetime64('2013-11-14 17:00:00'),
	'HarvardX/SW12.2x/1T2014': np.datetime64('2014-01-08 05:00:00'),
	'HarvardX/SW12.3x/1T2014': np.datetime64('2014-02-20 22:00:00'),
	'HarvardX/SW12.4x/1T2014': np.datetime64('2014-03-27 22:00:00'),
	'HarvardX/SW12.5x/2T2014': np.datetime64('2014-04-24 22:00:00'),
	'HarvardX/SW12.6x/2T2014': np.datetime64('2014-06-05 22:30:00'),
	'HarvardX/SW12.7x/3T2014': np.datetime64('2014-09-11 19:00:00'),
	'HarvardX/SW12.8x/3T2014': np.datetime64('2014-10-24 04:00:00'),
	'HarvardX/SW12.9x/3T2014': np.datetime64('2014-12-05 05:00:00'),
	'HarvardX/SW12.10x/1T2015': np.datetime64('2015-01-29 20:00:00'),
	#'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27 14:00:00'),
	#'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27 14:00:00'),
	#'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27 14:00:00'),
	#'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27 14:00:00'),
	#'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27 14:00:00'),
	#'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
	#'HarvardX/SW12.9x/2015': np.datetime64('2030-01-01'),
	#'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20')
}
# For each course:
#		Get demographic information from person-course dataset
#		Get list of rows from person-course-day dataset
#                 for all days between T0 and Tc; fill in any missing entries with 0s.
#	Predict certification using fixed number of observations (# days x # fields/day)
#	Compare to baseline predictors: P(certification | timeSinceLastAction) = logistic(- timeSinceLastAction)

def loadPersonCourseData ():
	d = pandas.io.parsers.read_csv('person_course_HarvardX_2015-11-11-051632.csv.gz', compression='gzip')
	d = convertTimes(d, 'start_time')
	return d

def loadPersonCourseDayData ():
	# Combine datasets
	d = pandas.io.parsers.read_csv('course_report_latest-person_course_day_SW12x.csv.gz', compression='gzip')
	e = pandas.io.parsers.read_csv('pcd_SW25x_1T2014.csv')
	d = pandas.concat((d, e))

	d = convertTimes(d, 'date')
	courseIds = np.unique(d.course_id)
	e = {}
	for courseId in courseIds:
		idxs = np.nonzero(d.course_id == courseId)[0]
		e[courseId] = d.iloc[idxs]
	return e

def makeVariable (shape, stddev, wd, name, collectionNames = [""]):
	var = tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)
	weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
	# Caller may wish to add to multiple collections
	for collectionName in collectionNames:
		tf.add_to_collection("losses{}".format(collectionName), weight_decay)
	return var

def runNN (train_x, train_y, test_x, test_y):
	global NUM_HIDDEN
	print "NN({})".format(NUM_HIDDEN)
	global LEARNING_RATE
	global MOMENTUM
	with tf.Graph().as_default():
		session = tf.InteractiveSession()

		x = tf.placeholder("float", shape=[None, train_x.shape[1]])
		y_ = tf.placeholder("float", shape=[None, train_y.shape[1]])

		W1 = makeVariable([train_x.shape[1],NUM_HIDDEN], stddev=0.5, wd=1e-1, name="W1")
		b1 = makeVariable([NUM_HIDDEN], stddev=0.5, wd=1e-1, name="b1")
		W2 = makeVariable([NUM_HIDDEN,train_y.shape[1]], stddev=0.5, wd=1e-1, name="W2")
		b2 = makeVariable([train_y.shape[1]], stddev=0.5, wd=1e-1, name="b2")

		level1 = tf.nn.relu(tf.matmul(x,W1) + b1)
		y = tf.nn.softmax(tf.matmul(level1,W2) + b2)

		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)), name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy)
		total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(LEARNING_RATE, batch * BATCH_SIZE, train_x.shape[0]/BATCH_SIZE, 0.95, staircase=True)
		train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(total_loss, global_step=batch)

		session.run(tf.initialize_all_variables())
		for i in range(NUM_EPOCHS):
			offset = i*BATCH_SIZE % (train_x.shape[0] - BATCH_SIZE)
			train_step.run({x: train_x[offset:offset+BATCH_SIZE, :], y_: train_y[offset:offset+BATCH_SIZE, :]})
			#if i % 500 == 0:
			#	auc = util.showProgress(cross_entropy, x, y, y_, train_x, train_y)
		auc = util.showProgress(cross_entropy, x, y, y_, test_x, test_y)
		session.close()
		return auc

def convertTimes (d, colName):
        goodIdxs = []
        idx = 0
        dates = []
        for dateStr in d[colName]:
                try:
                        date = np.datetime64(dateStr[0:10])  # "0:10" -- only extract the date
                        dates.append(date)
                        goodIdxs.append(idx)
                except:
                        pass
                idx += 1
        d = d.iloc[goodIdxs]
        dates = np.array(dates, dtype=np.datetime64)
        d[colName] = dates
        return d
		
def computeCourseDates (courseId):
	T0 = START_DATES[courseId]
	Tc = PREDICTION_DATES[courseId]
	#Tc = T0 + np.timedelta64(7, 'D')
	return T0, Tc

# Get users whose start_date is before Tc and who participated in course
def getRelevantUsers (pc, Tc):
	idxs = np.nonzero((pc.start_time < Tc) & (pc.viewed == 1))[0]
	return pc.username.iloc[idxs]

def getXandY (pc, pcd, usernames, T0, Tc, collapseOverTime = False, ignoreFirstWeek = False):
	# Restrict analysis to days between T0 and Tc
	idxs = np.nonzero((pcd.date >= T0) & (pcd.date < Tc))[0]
	pcd = pcd.iloc[idxs]
	
	# Create dummy variables
	pcUsernames = pc.username
	usernamesToCertifiedMap = { pcUsernames.iloc[i]:pc.certified.iloc[i] for i in range(len(pcUsernames)) }
	pcCertified = pc.certified
	DEMOGRAPHIC_FIELDS = [ 'continent', 'YoB', 'LoE', 'gender' ]
	pc = pc[DEMOGRAPHIC_FIELDS]
	pc = pandas.get_dummies(pc, columns = [ 'continent', 'LoE', 'gender' ])

	# For efficiency, figure out which rows of the person-course and person-course-day
	# datasets belong to which users
	usernamesToPcIdxsMap = dict(zip(pcUsernames, range(len(pc))))
	usernamesToPcdIdxsMap = {}
	for i in range(pcd.shape[0]):
		username = pcd.username.iloc[i]
		usernamesToPcdIdxsMap.setdefault(username, [])
		usernamesToPcdIdxsMap[username].append(i)

	# Only analyze users who appear in the person-course-day dataset
	usernames = list(set(usernames).intersection(usernamesToPcdIdxsMap.keys()))

	# Extract features for all users and put them into the design matrix X
	pcdDates = pcd.date
	pcd = pcd.drop([ 'username', 'course_id', 'date', 'last_event' ], axis=1)

	if ignoreFirstWeek:
		idxs = np.nonzero(pcdDates >= T0 + np.timedelta64(7, 'D'))[0]
		pcd.iloc[idxs] = 0

	# DEBUG -- Only test specific features
	#pcd = pcd[['nevents', 'sum_dt']]
	#pcd = pcd[['sum_dt']]
	# END DEBUG

	if collapseOverTime:
		NUM_DAYS = 1
	else:
		NUM_DAYS = int(math.ceil((Tc - T0) / np.timedelta64(1, 'D')))
	NUM_FEATURES = NUM_DAYS * len(pcd.columns) + len(pc.columns)
	X = np.zeros((len(usernames), NUM_FEATURES))
	y = np.zeros(len(usernames))
	sumDts = np.zeros((len(usernames), NUM_DAYS))  # Keep track of sum_dt as a special feature
	goodIdxs = []
	for i, username in enumerate(usernames):
		idxs = usernamesToPcdIdxsMap[username]
		# For each row in the person-course-day dataset for this user, put the
		# features into the correct column range for that user in the design matrix X.
		if collapseOverTime:
			X[i,0:len(pcd.columns)] = np.sum(pcd.iloc[idxs].as_matrix(), axis=0)  # Call as_matrix() so nan is treated as nan in sum!
			sumDts[i] = np.sum(pcd.sum_dt.iloc[idxs])
		else:
			for idx in idxs:
				dateIdx = int((np.datetime64(pcdDates.iloc[idx]) - T0) / np.timedelta64(1, 'D'))
				startColIdx = len(pcd.columns) * dateIdx
				sumDts[i,dateIdx] = pcd.sum_dt.iloc[idx]
				X[i,startColIdx:startColIdx+len(pcd.columns)] = pcd.iloc[idx]
		# Now append the demographic features
		demographics = pc.iloc[usernamesToPcIdxsMap[username]]
		X[i,NUM_DAYS * len(pcd.columns):] = demographics
		y[i] = usernamesToCertifiedMap[username]
		if np.isfinite(np.sum(X[i,:])):
			goodIdxs.append(i)
	return X[goodIdxs,:], y[goodIdxs], np.sum(sumDts[goodIdxs,:], axis=1)

def normalize (trainX, testX):
	mx = np.mean(trainX, axis=0)
	sx = np.std(trainX, axis=0)
	sx[sx == 0] = 1
	trainX -= np.tile(np.atleast_2d(mx), (trainX.shape[0], 1))
	trainX /= np.tile(np.atleast_2d(sx), (trainX.shape[0], 1))
	# Scale testing data using parameters estimated on training set
	testX -= np.tile(np.atleast_2d(mx), (testX.shape[0], 1))
	testX /= np.tile(np.atleast_2d(sx), (testX.shape[0], 1))
	return trainX, testX

def split (X, y, sumDts, trainIdxs = None, testIdxs = None):
	if trainIdxs == None:
		idxs = np.random.permutation(X.shape[0])
		numTraining = int(len(idxs) * 0.5)
		trainIdxs = idxs[0:numTraining]
		testIdxs = idxs[numTraining:]
	return X[trainIdxs,:], y[trainIdxs], sumDts[trainIdxs], X[testIdxs,:], y[testIdxs], sumDts[testIdxs], trainIdxs, testIdxs

def sampleWithReplacement (x, n):
	if len(x) == 0:
		return []
	idxs = (np.random.random(n) * len(x)).astype(np.int32)
	return x[idxs]

# Evaluate using bootstrapping to estimate estimate performance with a *uniform* distribution
# of sumDt for *both* classes.
def evaluateWithUniformSumDts (y, yhat, sumDts):
	sortedSumDts = np.sort(sumDts)
	pct1 = sortedSumDts[int(len(sumDts)*0.01)]
	pct99 = sortedSumDts[int(len(sumDts)*0.99)]
	NUM_CHUNKS = 20
	NUM_EXAMPLES_PER_CHUNK_PER_CLASS = 100
	chunkSize = (pct99 - pct1) / NUM_CHUNKS
	allY = []
	allYhat = []
	for sumDt in np.arange(pct1, pct99, chunkSize):
		sumDt1 = sumDt
		sumDt2 = sumDt + chunkSize
		posIdxs = np.nonzero((sumDts >= sumDt1) & (sumDts < sumDt2) & (y == 1))[0]
		negIdxs = np.nonzero((sumDts >= sumDt1) & (sumDts < sumDt2) & (y == 0))[0]
		posIdxs = sampleWithReplacement(posIdxs, NUM_EXAMPLES_PER_CHUNK_PER_CLASS)
		negIdxs = sampleWithReplacement(negIdxs, NUM_EXAMPLES_PER_CHUNK_PER_CLASS)
		allY += list(y[posIdxs]) + list(y[negIdxs])
		allYhat += list(yhat[posIdxs]) + list(yhat[negIdxs])
	return sklearn.metrics.roc_auc_score(allY, allYhat)

def splitAndGetNormalizedFeatures (somePc, somePcd, usernames, T0, Tc):
	# Get features and target values
	X, y, sumDts = getXandY(somePc, somePcd, usernames, T0, Tc, True, False)
	if len(np.nonzero(y == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(y == 1)[0]) < MIN_EXAMPLES:
		raise ValueError("Too few examples or all one class")
	# Split into training and testing folds
	trainX, trainY, trainSumDts, testX, testY, testSumDts, trainIdxs, testIdxs = split(X, y, sumDts)
	trainX, testX = normalize(trainX, testX)
	return trainX, trainY, testX, testY

def trainNN (trainX, trainY, testX, testY):
	global NUM_HIDDEN
	testY = np.atleast_2d(testY).T
	testY = np.hstack((1 - testY, testY))
	trainY = np.atleast_2d(trainY).T
	trainY = np.hstack((1 - trainY, trainY))
	return runNN(trainX, trainY, testX, testY)

def trainMLR (trainX, trainY, testX, testY):
	global MLR_REG
	baselineModel = sklearn.linear_model.LogisticRegression(C=MLR_REG)
	baselineModel.fit(trainX, trainY)
	yhat = baselineModel.predict_proba(testX)[:,1]
	aucMLR = sklearn.metrics.roc_auc_score(testY, yhat)
	return aucMLR

def prepareAllData (pc, pcd):
	print "Preparing data..."
	allCourseData = {}
	for courseId in set(pcd.keys()).intersection(START_DATES.keys()):  # For each course
		print courseId
		# Restrict analysis to rows of PC dataset relevant to this course
		idxs = np.nonzero(pc.course_id == courseId)[0]
		somePc = pc.iloc[idxs]
		idxs = np.nonzero(pcd[courseId].course_id == courseId)[0]
		somePcd = pcd[courseId].iloc[idxs]
		T0, Tc = computeCourseDates(courseId)
		usernames = getRelevantUsers(somePc, Tc)
		allData = splitAndGetNormalizedFeatures (somePc, somePcd, usernames, T0, Tc)
		allCourseData[courseId] = allData
	print "...done"
	return allCourseData

def runExperiments (allCourseData, useNN):
	allAucs = []
	for courseId in set(pcd.keys()).intersection(START_DATES.keys()):  # For each course
		# Find start date T0 and cutoff date Tc
		(trainX, trainY, testX, testY) = allCourseData[courseId]
		if useNN:
			allAucs.append(trainNN(trainX, trainY, testX, testY))
		else:
			allAucs.append(trainMLR(trainX, trainY, testX, testY))
	return np.mean(allAucs)

def optimize (allCourseData):
	# MLR
	MLR_REG_SET = 10. ** np.arange(-5, +6).astype(np.float32)
	bestAuc = -1
	for paramValue in MLR_REG_SET:
		global MLR_REG
		MLR_REG = paramValue
		avgAuc = runExperiments(allCourseData, False)
		print avgAuc
		if avgAuc > bestAuc:
			bestAuc = avgAuc
			bestParamValue = paramValue
	print "MLR: {} for {}".format(bestAuc, bestParamValue)

	# NN
	global NUM_HIDDEN
	NUM_HIDDEN = 2
	global LEARNING_RATE
	LEARNING_RATE_SET = 10. ** np.arange(-4, 0, 0.5).astype(np.float32)
	global MOMENTUM
	MOMENTUM_SET = 10. ** np.arange(-4, 0, 0.5).astype(np.float32)
	global NUM_EPOCHS
	NUM_EPOCHS_SET = np.arange(1000, 11000, 1000).astype(np.float32)
	for learningRate in LEARNING_RATE_SET:
		for momentum in MOMENTUM_SET:
			for numEpochs in NUM_EPOCHS_SET:
				LEARNING_RATE = learningRate
				MOMENTUM = momentum
				NUM_EPOCHS = numEpochs
				avgAuc = runExperiments(allCourseData, True)
				print avgAuc
				if avgAuc > bestAuc:
					bestAuc = avgAuc
					bestParamValue = (learningRate, momentum, numEpochs)
	print "NN: {} for {}".format(bestAuc, bestParamValue)

if __name__ == "__main__":
	if 'pcd' not in globals():
		pcd = loadPersonCourseDayData()
		pc = loadPersonCourseData()
	if 'allCourseData' not in globals():
		allCourseData = prepareAllData(pc, pcd)
	optimize(allCourseData)
