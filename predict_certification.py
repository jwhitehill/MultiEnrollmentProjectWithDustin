import util
import pandas
import cPickle
import pandas
import math
import numpy as np
import sklearn.metrics
import sklearn.linear_model
from common import loadData, getCourseStartAndEndDates, NUM_WEEKS_HEURISTIC, convertFieldsToDummies

BATCH_SIZE = 100
WEEK = np.timedelta64(7, 'D')
MIN_EXAMPLES = 10

def loadCourseDates ():
	d = pandas.io.parsers.read_csv('pts_accumulation_table.csv')
	startDates = dict(zip(d.course_id, d['time_pts0.0'].astype(np.datetime64)))
	predictionDates0_5 = dict(zip(d.course_id, d['time_pts0.5'].astype(np.datetime64)))
	predictionDates1_0 = dict(zip(d.course_id, d['time_pts1.0'].astype(np.datetime64)))
	return startDates, predictionDates0_5, predictionDates1_0

START_DATES, PREDICTION_DATES_0_5, PREDICTION_DATES_1_0 = loadCourseDates()

#START_DATES = {
#	'HarvardX/SW25x/1T2014': np.datetime64('2014-02-25'),
#	'HarvardX/SW12x/2013_SOND': np.datetime64('2013-10-31'),
#	'HarvardX/SW12.2x/1T2014': np.datetime64('2014-01-02'),
#	'HarvardX/SW12.3x/1T2014': np.datetime64('2014-02-13'),
#	'HarvardX/SW12.4x/1T2014': np.datetime64('2014-03-20'),
#	'HarvardX/SW12.5x/2T2014': np.datetime64('2014-04-24'),
#	'HarvardX/SW12.6x/2T2014': np.datetime64('2014-05-22'),
#	'HarvardX/SW12.7x/3T2014': np.datetime64('2014-09-04'),
#	'HarvardX/SW12.8x/3T2014': np.datetime64('2014-10-09'),
#	'HarvardX/SW12.9x/3T2014': np.datetime64('2014-11-20'),
#	'HarvardX/SW12.10x/1T2015': np.datetime64('2015-01-08'),
#	'HarvardX/PH231x/1T2016': np.datetime64('2016-01-25'),
#	'HarvardX/PH557/3T2015': np.datetime64('2015-12-03'),
#	'HarvardX/PH525.4x/3T2015': np.datetime64('2016-01-15'),
#	'HarvardX/MUS24.3x/1T2016': np.datetime64('2016-01-21'),
#	#'HarvardX/PH556/2015T3': np.datetime64('2016-01-20'),
#	'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27'),
#	'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27'),
#	'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27'),
#	'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27'),
#	'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27'),
#	'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
#	#'HarvardX/SW12.9x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20'),
#}

# Dates corresponding to when students can earn 0.5 * number of points necessary for certification
#PREDICTION_DATES_0_5 = {
#	'HarvardX/SW25x/1T2014': np.datetime64('2014-03-18 17:00:00'),
#	'HarvardX/SW12x/2013_SOND': np.datetime64('2013-11-14 17:00:00'),
#	'HarvardX/SW12.2x/1T2014': np.datetime64('2014-01-08 05:00:00'),
#	'HarvardX/SW12.3x/1T2014': np.datetime64('2014-02-20 22:00:00'),
#	'HarvardX/SW12.4x/1T2014': np.datetime64('2014-03-27 22:00:00'),
#	'HarvardX/SW12.5x/2T2014': np.datetime64('2014-04-24 22:00:00'),
#	'HarvardX/SW12.6x/2T2014': np.datetime64('2014-06-05 22:30:00'),
#	'HarvardX/SW12.7x/3T2014': np.datetime64('2014-09-11 19:00:00'),
#	'HarvardX/SW12.8x/3T2014': np.datetime64('2014-10-24 04:00:00'),
#	'HarvardX/SW12.9x/3T2014': np.datetime64('2014-12-05 05:00:00'),
#	'HarvardX/SW12.10x/1T2015': np.datetime64('2015-01-29 20:00:00'),
#	'HarvardX/PH231x/1T2016': np.datetime64('2016-02-29'),
#	'HarvardX/PH557/3T2015': np.datetime64('2016-02-29'),
#	'HarvardX/PH525.4x/3T2015': np.datetime64('2016-02-29'),
#	'HarvardX/MUS24.3x/1T2016': np.datetime64('2016-02-29'),
#	#'HarvardX/PH556/2015T3': np.datetime64('2016-02-29'),
#	'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
#	#'HarvardX/SW12.9x/2015': np.datetime64('2030-01-01'),
#	'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20')
#}

	
# Dates corresponding to when students can earn 1.0 * number of points necessary for certification
#PREDICTION_DATES_1_0 = {
#	'HarvardX/SW25x/1T2014':      np.datetime64('2014-04-02 00:00:00'),
#	'HarvardX/SW12x/2013_SOND':   np.datetime64('2013-12-05 21:00:00'),
#	'HarvardX/SW12.2x/1T2014':    np.datetime64('2014-01-23 17:30:00'),
#	'HarvardX/SW12.3x/1T2014':    np.datetime64('2014-02-27 22:00:00'),
#	'HarvardX/SW12.4x/1T2014':    np.datetime64('2014-04-10 18:00:00'),
#	'HarvardX/SW12.5x/2T2014':    np.datetime64('2014-05-08 19:00:00'),
#	'HarvardX/SW12.6x/2T2014':    np.datetime64('2014-06-19 20:30:00'),
#	'HarvardX/SW12.7x/3T2014':    np.datetime64('2014-09-25 20:00:00'),
#	'HarvardX/SW12.8x/3T2014':    np.datetime64('2014-11-06 20:00:00'),
#	'HarvardX/SW12.9x/3T2014':    np.datetime64('2014-12-19 02:30:00'),
#	'HarvardX/SW12.10x/1T2015':   np.datetime64('2015-02-27 02:00:00'),
#	'HarvardX/PH231x/1T2016': np.datetime64('2016-02-29'),
#	'HarvardX/PH557/3T2015': np.datetime64('2016-02-29'),
#	'HarvardX/PH525.4x/3T2015': np.datetime64('2016-02-29'),
#	'HarvardX/MUS24.3x/1T2016': np.datetime64('2016-02-29'),
#	#'HarvardX/PH556/2015T3': np.datetime64('2016-02-29'),
#	'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27 14:00:00'),
#	'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
#	'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
#	#'HarvardX/SW12.9x/2015': np.datetime64('2030-01-01'),
#	'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20')
#}

# For each course:
#		Get demographic information from person-course dataset
#		Get list of rows from person-course-day dataset
#                 for all days between T0 and Tc; fill in any missing entries with 0s.
#	Predict certification using fixed number of observations (# days x # fields/day)
#	Compare to baseline predictors: P(certification | timeSinceLastAction) = logistic(- timeSinceLastAction)

def loadPersonCourseData ():
	#d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_jwaldo/BigQuery/person_course_HarvardX_2015-11-11-051632.csv')
	d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_charlesriverx/HarvardX/CoursesAll/person_course.csv')
	d = convertTimes(d, 'start_time')
	return d

def loadPrecourseSurveyData ():
	d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_charlesriverx/HarvardX/CoursesAll/person_course_survey_latest.csv')
	return d

def loadPersonCourseDayData ():
	# Combine datasets
	d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_charlesriverx/HarvardX/CoursesAll/person_course_day.csv')

	d = convertTimes(d, 'date')
	courseIds = np.unique(d.course_id)
	e = {}
	for courseId in courseIds:
		idxs = np.nonzero(d.course_id == courseId)[0]
		e[courseId] = d.iloc[idxs]
	return e

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
		
def computeCourseDates (courseId, startDates):
	T0 = startDates[courseId]
	Tc = PREDICTION_DATES_1_0[courseId]
	return T0, Tc

# Get users whose start_date is before Tc and who participated in course
def getRelevantUsers (pc, Tc):
	idxs = np.nonzero((pc.start_time < Tc) & (pc.viewed == 1))[0]
	return pc.username.iloc[idxs]

def convertYoB (YoB):
	REF_YEAR = 2012
	ages = REF_YEAR - YoB
	ageRanges = [ -float('inf'), 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, +float('inf') ]
	newYoB = np.zeros_like(ages)
	for i in range(len(ageRanges) - 1):
		minAge = ageRanges[i]
		maxAge = ageRanges[i+1]
		idxs = np.nonzero((ages >= minAge) & (ages < maxAge))
		# In code below, we add 1 ("+ 1") so that the minimum index corresponding to
		# any valid age range is 1, not 0. It follows that any invalid age range (i.e., NaN)
		# will retain value 0.
		newYoB[idxs] = i + 1 
	return newYoB

def getXandY (pc, pcd, usernames, T0, Tc, demographicsOnly):
	# Restrict analysis to days between T0 and Tc
	idxs = np.nonzero((pcd.date >= T0) & (pcd.date < Tc))[0]
	pcd = pcd.iloc[idxs]
	
	# Create dummy variables
	pcUsernames = pc.username
	usernamesToCertifiedMap = { pcUsernames.iloc[i]:pc.certified.iloc[i] for i in range(len(pcUsernames)) }
	usernamesToLastEventMap = { pcUsernames.iloc[i]:pc.last_event.iloc[i] for i in range(len(pcUsernames)) }
	DEMOGRAPHIC_FIELDS = [ 'continent', 'YoB', 'LoE', 'gender' ]
	pc = pc[DEMOGRAPHIC_FIELDS]
	pc.YoB = convertYoB(pc.YoB)
	#pc = pandas.get_dummies(pc, columns = [ 'continent', 'LoE', 'gender', 'YoB' ], dummy_na = True)
	pc = convertFieldsToDummies(pc)
	for field in ['continent', 'YoB', 'LoE', 'gender']:
		if np.sum(pc[field].isnull()) > 0:
			1/0

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
	pcdDates = pcd.date
	pcd = pcd.drop([ 'username', 'course_id', 'date', 'last_event' ], axis=1)

	# Convert NaNs in person-course-day dataset to 0
	pcd = pcd.fillna(value=0)

	NUM_DAYS = 1
	NUM_FEATURES = NUM_DAYS * len(pcd.columns) + len(pc.columns)
	X = np.zeros((len(usernames), NUM_FEATURES))
	Xheur = np.zeros(len(usernames))
	y = np.zeros(len(usernames))
	sumDts = np.zeros((len(usernames), NUM_DAYS))  # Keep track of sum_dt as a special feature
	goodIdxs = []
	for i, username in enumerate(usernames):
		if username in usernamesToPcdIdxsMap.keys():
			idxs = usernamesToPcdIdxsMap[username]
			# For each row in the person-course-day dataset for this user, put the
			# features into the correct column range for that user in the design matrix X.
			X[i,0:len(pcd.columns)] = np.sum(pcd.iloc[idxs].as_matrix(), axis=0)  # Call as_matrix() so nan is treated as nan in sum!
			sumDts[i] = np.sum(pcd.sum_dt.iloc[idxs])
		else:
			X[i,0:len(pcd.columns)] = np.zeros(len(pcd.columns))
			sumDts[i] = 0
		# Now append the demographic features
		demographics = pc.iloc[usernamesToPcIdxsMap[username]]
		X[i,NUM_DAYS * len(pcd.columns):] = demographics
		# "Heuristic" predictor -- whether the student's last event time is before/after the first week of the course
		lastEvent = usernamesToLastEventMap[username]
		if (lastEvent != 'nan') and (lastEvent == lastEvent):  # np.isfinite doesn't work for dates, so we have to check if it equals itself
			#Xheur[i] = np.datetime64(lastEvent[0:10]) > (T0 + np.timedelta64(7*NUM_WEEKS_HEURISTIC, 'D'))  # Did they persist beyond 2 weeks into course?
			Xheur[i] = np.datetime64(lastEvent[0:10]) >= (Tc - np.timedelta64(7, 'D'))  # Any action within last week?
		else:
			Xheur[i] = 0
		y[i] = usernamesToCertifiedMap[username]
		if np.isfinite(np.sum(X[i,:])):
			goodIdxs.append(i)
	
	if demographicsOnly:
		X[:,0:len(pcd.columns)] = 0  # Zero out the non-demographics information
	return X[goodIdxs,:], Xheur[goodIdxs], y[goodIdxs], np.sum(sumDts[goodIdxs,:], axis=1)

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

def split (X, Xheur, y, sumDts, trainIdxs = None, testIdxs = None):
	if trainIdxs == None:
		idxs = np.random.permutation(X.shape[0])
		numTraining = int(len(idxs) * 0.5)
		trainIdxs = idxs[0:numTraining]
		testIdxs = idxs[numTraining:]
	return X[trainIdxs,:], Xheur[trainIdxs], y[trainIdxs], sumDts[trainIdxs], X[testIdxs,:], Xheur[testIdxs], y[testIdxs], sumDts[testIdxs], trainIdxs, testIdxs

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

def splitAndGetNormalizedFeatures (somePc, somePcd, usernames, T0, Tc, demographicsOnly):
	# Get features and target values
	X, Xheur, y, sumDts = getXandY(somePc, somePcd, usernames, T0, Tc, demographicsOnly)
	if len(np.nonzero(y == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(y == 1)[0]) < MIN_EXAMPLES:
		raise ValueError("Too few examples or all one class")
	# Split into training and testing folds
	trainX, trainXheur, trainY, trainSumDts, testX, testXheur, testY, testSumDts, trainIdxs, testIdxs = split(X, Xheur, y, sumDts)
	trainX, testX = normalize(trainX, testX)
	return trainX, trainXheur, trainY, testX, testXheur, testY

def trainMLR (trainX, trainY, testX, testY, mlrReg):
	baselineModel = sklearn.linear_model.LogisticRegression(C=mlrReg)
	baselineModel.fit(trainX, trainY)
	yhat = baselineModel.predict_proba(testX)[:,1]
	aucMLR = sklearn.metrics.roc_auc_score(testY, yhat)
	return baselineModel, aucMLR, (testY, yhat)

def prepareAllData (startDates, endDates, demographicsOnly):
	print "Preparing data..."
	allCourseData = {}
	#for courseId in set(pcd.keys()).intersection(START_DATES.keys()):  # For each course
	for courseId in set(startDates.keys()).intersection(START_DATES.keys()):  # For each course
		# Load data for this course
		print "Loading {}...".format(courseId)
		try:
			somePc, _, somePcd = loadData(courseId)
		except (IOError, pandas.io.parsers.EmptyDataError):
			print "Skipping"
			continue
		# If no certifiers, then skip
		if (np.sum(somePc.certified) < MIN_EXAMPLES) or (np.sum(somePc.certified) >= len(somePc) - MIN_EXAMPLES):
			print "Skipping"
			continue

		T0, Tc = computeCourseDates(courseId, startDates)
		allCourseData[courseId] = []
		print "...done"

		Tcutoffs = np.arange(T0 + 1*WEEK, Tc+np.timedelta64(1, 'D'), WEEK)
		for Tcutoff in Tcutoffs:
			usernames = getRelevantUsers(somePc, Tcutoff)
			allData = splitAndGetNormalizedFeatures(somePc, somePcd, usernames, T0, Tcutoff, demographicsOnly)
			allCourseData[courseId].append(allData)
	print "...done"
	return allCourseData

def runExperimentsHeuristic ():
	allAucs = {}
	for courseId in set(allCourseData.keys()).intersection(START_DATES.keys()):  # For each course
		allAucs[courseId] = []
		for i, weekData in enumerate(allCourseData[courseId]):
			if i >= (NUM_WEEKS_HEURISTIC - 1):
				(trainX, trainXheur, trainY, testX, testXheur, testY) = weekData
				auc = sklearn.metrics.roc_auc_score(testY, testXheur)
				allAucs[courseId].append(auc)
	return allAucs

def compareToCrossTrain (courseId, testX, testY, weekIdxWRTTc):
	discipline = COURSE_TO_DISCIPLINE_MAP[courseId]
	(xtrainCourseId, modelList) = PRETRAINED_MODELS[discipline]
	if xtrainCourseId == courseId:
		return float('nan')  # Not allowed to "cross-train" on same course!

	# Select model that corresponds most closely in time (relative to T_c) to specified weekIdx
	if weekIdxWRTTc < len(modelList):
		idx = len(modelList) - weekIdxWRTTc - 1
	else:
		idx = 0  # As far back in time as we can go
	model = modelList[idx]
	acc = sklearn.metrics.roc_auc_score(testY, model.predict_proba(testX)[:,1])
	print "Crosstrain ", courseId, acc
	return acc

def runExperiments (allCourseData):
	allAucs = {}
	allCrosstrainAucs = {}
	allDists = {}
	models = {}
	for courseId in set(allCourseData.keys()).intersection(START_DATES.keys()):  # For each course
		print courseId
		allAucs[courseId] = []
		allCrosstrainAucs[courseId] = []
		allDists[courseId] = []
		models[courseId] = []
		for i, weekData in enumerate(allCourseData[courseId]):
			(trainX, trainXheur, trainY, testX, testXheur, testY) = weekData
			global MLR_REG
			print MLR_REG
			model, auc, dist = trainMLR(trainX, trainY, testX, testY, MLR_REG)
			weekIdxWRTTc = len(allCourseData[courseId]) - i - 1
			allCrosstrainAucs[courseId].append(compareToCrossTrain(courseId, testX, testY, weekIdxWRTTc))
			allAucs[courseId].append(auc)
			allDists[courseId].append(dist)
			models[courseId].append(model)
	return allAucs, allCrosstrainAucs, allDists, models

def trainAllHeuristic ():
	allAucs = runExperimentsHeuristic()
	cPickle.dump(allAucs, open("results_heuristic.pkl", "wb"))

def trainAll (allCourseData, demographicsOnly, save=True):
	allAucs, allCrosstrainAucs, allDists, allModels = runExperiments(allCourseData)
	if save:
		cPickle.dump(allAucs, open("results_prong1{}.pkl".format("_demog" if demographicsOnly else ""), "wb"))
		cPickle.dump(allCrosstrainAucs, open("results_xtrain_prong1{}.pkl".format("_demog" if demographicsOnly else ""), "wb"))
		cPickle.dump(allDists, open("results_prong1_dists.pkl", "wb"))
		cPickle.dump(allModels, open("results_prong1_models.pkl", "wb"))
	return allAucs

def optimize (allCourseData):
	MLR_REG_SET = 10. ** np.arange(-5, +6).astype(np.float32)
	bestAuc = -1
	for paramValue in MLR_REG_SET:
		global MLR_REG
		MLR_REG = float(paramValue)
		allAucs, = runExperiments(allCourseData)
		avgAuc = np.mean(aucs.values())
		print avgAuc
		if avgAuc > bestAuc:
			bestAuc = avgAuc
			bestParamValue = paramValue
	print "Accuracy: {} for {}".format(bestAuc, bestParamValue)

def loadCourseToDisciplineMap ():
	d = pandas.read_csv('course_to_discipline.csv')
	return { d.course_id.iloc[i]:d.discipline_grouping.iloc[i] for i in range(len(d)) }

def loadPretrainedModels ():
	models = cPickle.load(open('results_prong1_models.pkl', 'rb'))
	# These are courses from different disciplines that were among top 20 HX courses with most certifying participants
	courseIds = [ 'HarvardX/GSE2x/2T2014', 'HarvardX/PH525.1x/1T2015', 'HarvardX/SPU30x/2T2014', 'HarvardX/AmPoX.4/1T2015' ]
	theMap = { COURSE_TO_DISCIPLINE_MAP[courseId]:(courseId, models[courseId]) for courseId in courseIds }
	return theMap

if __name__ == "__main__":
	COURSE_TO_DISCIPLINE_MAP = loadCourseToDisciplineMap()
	PRETRAINED_MODELS = loadPretrainedModels()

	DEMOGRAPHICS_ONLY = False
	if 'startDates' not in globals():
		startDates, endDates = getCourseStartAndEndDates()
	if 'allCourseData' not in globals():
		allCourseData = prepareAllData(startDates, endDates, DEMOGRAPHICS_ONLY)
	#optimize(allCourseData)
	MLR_REG = 1.
	allAucs = trainAll(allCourseData, DEMOGRAPHICS_ONLY, save=True)

	trainAllHeuristic()
