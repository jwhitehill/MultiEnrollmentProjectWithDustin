import util
import pandas
import math
import numpy as np
import sklearn.metrics
import sklearn.linear_model
from predict_persistence import loadPersonCourseData, loadPersonCourseDayData, \
                                convertTimes, getRelevantUsers, convertYoB, normalize, \
				trainMLR


# Dates corresponding to when students can earn 0.5 * number of points necessary for certification
PREDICTION_DATES_1_0 = {
	'HarvardX/SW25x/1T2014':      np.datetime64('2014-04-02 00:00:00'),
	'HarvardX/SW12x/2013_SOND':   np.datetime64('2013-12-05 21:00:00'),
	'HarvardX/SW12.2x/1T2014':    np.datetime64('2014-01-23 17:30:00'),
	'HarvardX/SW12.3x/1T2014':    np.datetime64('2014-02-27 22:00:00'),
	'HarvardX/SW12.4x/1T2014':    np.datetime64('2014-04-10 18:00:00'),
	'HarvardX/SW12.5x/2T2014':    np.datetime64('2014-05-08 19:00:00'),
	'HarvardX/SW12.6x/2T2014':    np.datetime64('2014-06-19 20:30:00'),
	'HarvardX/SW12.7x/3T2014':    np.datetime64('2014-09-25 20:00:00'),
	'HarvardX/SW12.8x/3T2014':    np.datetime64('2014-11-06 20:00:00'),
	'HarvardX/SW12.9x/3T2014':    np.datetime64('2014-12-19 02:30:00'),
	'HarvardX/SW12.10x/1T2015':   np.datetime64('2015-02-27 02:00:00')
}

def computeCourseDates (courseId):
	T0 = START_DATES[courseId]
	Tc_1_0 = PREDICTION_DATES_1_0[courseId]
	return T0, Tc_1_0

def runExperiments (allCourseData):
	allAucs = []
	for courseId in set(pcd.keys()).intersection(START_DATES.keys()):  # For each course
		# Find start date T0 and cutoff date Tc
		(trainX, trainY, testX, testY) = allCourseData[courseId]
		auc = trainMLR(trainX, trainY, testX, testY)
		allAucs.append(auc)
	return np.mean(allAucs)

def optimize (allCourseData):
	MLR_REG_SET = 10. ** np.arange(-5, +6).astype(np.float32)
	bestAuc = -1
	for paramValue in MLR_REG_SET:
		global MLR_REG
		MLR_REG = float(paramValue)
		avgAuc = runExperiments(allCourseData, predictCertification)
		print avgAuc
		if avgAuc > bestAuc:
			bestAuc = avgAuc
			bestParamValue = paramValue
	print "Accuracy: {} for {}".format(bestAuc, bestParamValue)

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

	# Only analyze users who appear in the person-course-day dataset
	usernames = list(set(usernames).intersection(usernamesToPcdIdxsMap.keys()))

	# Extract features for all users and put them into the design matrix X
	pcdDates = pcd.date
	pcd = pcd.drop([ 'username', 'course_id', 'date', 'last_event' ], axis=1)

	if ignoreFirstWeek:
		idxs = np.nonzero(pcdDates >= T0 + np.timedelta64(7, 'D'))[0]
		pcd.iloc[idxs] = 0

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

def splitAndGetNormalizedFeatures (somePc, somePcd, usernames, T0, Tc):
	# Get features and target values
	trainX, trainY, testX, testY = getXandY(somePc, somePcd, usernames, T0, Tc, True, False)
	if len(np.nonzero(y == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(y == 1)[0]) < MIN_EXAMPLES:
		raise ValueError("Too few examples or all one class")
	return trainX, trainY, testX, testY

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
		allData = splitAndGetNormalizedFeatures(somePc, somePcd, usernames, T0, Tc)
		allCourseData[courseId] = allData
	print "...done"
	return allCourseData

if __name__ == "__main__":
	if 'pcd' not in globals():
		pcd = loadPersonCourseDayData()
		pc = loadPersonCourseData()
	if 'allCourseData' not in globals():
		allCourseData = prepareAllData(pc, pcd)
	optimize(allCourseData)
