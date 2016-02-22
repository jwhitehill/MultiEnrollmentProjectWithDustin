import util
import cPickle
import quantify
import pandas
import math
import numpy as np
import sklearn.metrics
import sklearn.linear_model
from predict_certification import loadPersonCourseData, loadPersonCourseDayData, \
                                convertTimes, getRelevantUsers, convertYoB, \
				trainMLR, START_DATES, MIN_EXAMPLES, WEEK, PREDICTION_DATES_1_0 

def computeCourseDates (courseId):
	T0 = START_DATES[courseId]
	Tc_1_0 = PREDICTION_DATES_1_0[courseId]
	return T0, Tc_1_0

def runExperiments (allCourseData):
	allAucs = {}
	allAucsCert = {}
	for courseId in set(pcd.keys()).intersection(START_DATES.keys()):  # For each course
		#print courseId
		allAucs[courseId] = []
		allAucsCert[courseId] = []
		for i, weekData in enumerate(allCourseData[courseId]):
			# Find start date T0 and cutoff date Tc
			(trainX, trainY, trainYcert, testX, testY, testYcert) = weekData
			auc, _ = trainMLR(trainX, trainY, testX, testY, 1.)
			aucCert, _ = trainMLR(trainX, trainY, testX, testYcert, 1.)
			#print "To predict week {}: {}".format(i+3, auc)
			allAucs[courseId].append(auc)
			allAucsCert[courseId].append(aucCert)
		#print
	return allAucs, allAucsCert

def trainAll (allCourseData):
	global MLR_REG
	MLR_REG = 1.
	results = runExperiments(allCourseData)
	cPickle.dump(results, open("results_prong2.pkl", "wb"))

def optimize (allCourseData):
	MLR_REG_SET = 10. ** np.arange(-5, +6).astype(np.float32)
	bestAuc = -1
	for paramValue in MLR_REG_SET:
		global MLR_REG
		MLR_REG = float(paramValue)
		allAucs, unused = runExperiments(allCourseData)
		avgAuc = np.mean(np.hstack(allAucs.values()))
		print allAucs
		print "Mean acc: {}".format(avgAuc)
		if avgAuc > bestAuc:
			bestAuc = avgAuc
			bestParamValue = paramValue
	print "Accuracy: {} for {}".format(bestAuc, bestParamValue)

#def convertToQuantiles (X):
#	X = np.array(X, dtype=np.float32)
#	N = X.shape[0]
#	Xquantiles = np.zeros_like(X, dtype=np.float32)
#	for i in range(X.shape[1]):
#		col = X[:,i]
#		colSorted = np.tile(np.atleast_2d(np.sort(col, axis=0)), (N, 1))
#		colRep = np.tile(np.atleast_2d(col).T, (1, N))
#		# Line below: for each observation (element of col), find the smallest index
#		# in the *sorted* column that is >= that observation. Then normalize.
#		Xquantiles[:,i] = np.argmax(colRep <= colSorted, axis=1) / float(N)
#	return Xquantiles

def getXandY (pc, pcd, usernames, T0, Tc, normalize):
	# TARGET VALUES
	# The target value for each user consists of whether or not the user
	# did *anything* during the week just prior to Tc
	idxs = np.nonzero((pcd.date >= Tc - WEEK) & (pcd.date < Tc))[0]
	lastWeekPcd = pcd.iloc[idxs]
	grouping = lastWeekPcd.groupby('username')
	lastWeekUsernames = np.array(grouping.groups.keys())
	persistenceIdxs = np.nonzero(grouping.sum_dt.sum() > 0)[0]
	usersWhoPersisted = set(lastWeekUsernames[persistenceIdxs])

	# FEATURE EXTRACTION
	# Restrict analysis to days between T0 and Tc-WEEK
	idxs = np.nonzero((pcd.date >= T0) & (pcd.date < Tc - WEEK))[0]
	pcd = pcd.iloc[idxs]
	
	# Create dummy variables
	pcUsernames = pc.username
	usernamesToCertifiedMap = { pcUsernames.iloc[i]:pc.certified.iloc[i] for i in range(len(pcUsernames)) }
	DEMOGRAPHIC_FIELDS = [ 'continent', 'YoB', 'LoE', 'gender' ]
	pc = pc[DEMOGRAPHIC_FIELDS]
	pc.YoB = convertYoB(pc.YoB)
	pc = pandas.get_dummies(pc, columns = [ 'continent', 'LoE', 'gender', 'YoB' ], dummy_na = True)

	# For efficiency, figure out which rows of the person-course and person-course-day
	# datasets belong to which users
	usernamesToPcIdxsMap = dict(zip(pcUsernames, range(len(pc))))
	usernamesToPcdIdxsMap = {}
	for i in range(pcd.shape[0]):
		username = pcd.username.iloc[i]
		usernamesToPcdIdxsMap.setdefault(username, [])
		usernamesToPcdIdxsMap[username].append(i)

	### Only analyze users who appear in the person-course-day dataset
	##usernames = list(set(usernames).intersection(usernamesToPcdIdxsMap.keys()))

	# Extract features for all users and put them into the design matrix X
	pcd = pcd.drop([ 'username', 'course_id', 'date', 'last_event' ], axis=1)
	
	# Convert NaNs in person-course-day dataset to 0
	pcd = pcd.fillna(value=0)
	pcd = pcd.as_matrix()
	if normalize:
		pcd = pcd.astype(np.float32)
		quantify.quantify(pcd.shape[0], pcd.shape[1], pcd)

	NUM_FEATURES = pcd.shape[1] + len(pc.columns)
	X = np.zeros((len(usernames), NUM_FEATURES))
	y = np.zeros(len(usernames))
	yCert = np.zeros(len(usernames))
	for i, username in enumerate(usernames):
		if username in usernamesToPcdIdxsMap.keys():
			idxs = usernamesToPcdIdxsMap[username]
			# For each row in the person-course-day dataset for this user, put the
			# features into the correct column range for that user in the design matrix X.
			X[i,0:pcd.shape[1]] = np.sum(pcd[idxs,:], axis=0)
		else:
			X[i,0:pcd.shape[1]] = np.zeros(pcd.shape[1])
		# Now append the demographic features
		demographics = pc.iloc[usernamesToPcIdxsMap[username]]
		X[i,pcd.shape[1]:] = demographics
		y[i] = username in usersWhoPersisted
		yCert[i] = usernamesToCertifiedMap[username]
	return X, y, yCert

def extractFeaturesAndTargets (somePc, somePcd, usernames, T0, Tc, normalize):
	# Get features and target values
	trainX, trainY, trainYcert = getXandY(somePc, somePcd, usernames, T0, Tc - 1*WEEK, normalize)
	testX, testY, testYcert = getXandY(somePc, somePcd, usernames, T0, Tc, normalize)

	if len(np.nonzero(trainY == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(trainY == 1)[0]) < MIN_EXAMPLES:
		raise ValueError("Train: Too few examples or all one class")
	if len(np.nonzero(testY == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(testY == 1)[0]) < MIN_EXAMPLES:
		raise ValueError("Test: Too few examples or all one class")
	return trainX, trainY, trainYcert, testX, testY, testYcert

def prepareAllData (pc, pcd, normalize):
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
		allCourseData[courseId] = []
		# We need at least 3 weeks' worth of data to both train and test the model.
		# We use the first 2 weeks' data to train a model (labels are determined by week 2, and
		# features are extracted from week 1). But then to *evaluate* that model, we need
		# another (3rd) week.
		Tcutoffs = np.arange(T0 + 3*WEEK, Tc, WEEK)
		print courseId, Tcutoffs
		for Tcutoff in Tcutoffs:
			# The users that we train/test on must have entered the course by the end of the
			# *first* week of the last 3 weeks in the time range. Hence, we subtract 2 weeks.
			usernames = getRelevantUsers(somePc, Tcutoff - 2*WEEK)
			allData = extractFeaturesAndTargets(somePc, somePcd, usernames, T0, Tcutoff, normalize)
			allCourseData[courseId].append(allData)
	print "...done"
	return allCourseData

if __name__ == "__main__":
	NORMALIZE = True
	if 'pcd' not in globals():
		pcd = loadPersonCourseDayData()
		pc = loadPersonCourseData()
	if 'allCourseData' not in globals():
		allCourseData = prepareAllData(pc, pcd, NORMALIZE)
	#optimize(allCourseData)
	trainAll(allCourseData)
