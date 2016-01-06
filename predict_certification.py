import pandas
import numpy as np
import sklearn.metrics
import sklearn.linear_model

MIN_EXAMPLES = 10
START_DATES = {
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
	'HarvardX/SW12.1x/2015': np.datetime64('2015-10-27'),
	'HarvardX/SW12.2x/2015': np.datetime64('2015-10-27'),
	'HarvardX/SW12.3x/2015': np.datetime64('2015-10-27'),
	'HarvardX/SW12.4x/2015': np.datetime64('2015-10-27'),
	'HarvardX/SW12.5x/2015': np.datetime64('2015-10-27'),
	'HarvardX/SW12.6x/2015': np.datetime64('2015-11-20'),
	'HarvardX/SW12.7x/2015': np.datetime64('2015-11-20'),
	'HarvardX/SW12.8x/2015': np.datetime64('2015-11-20'),
	'HarvardX/SW12.9x/2015': np.datetime64('2015-11-20'),
	'HarvardX/SW12.10x/2015': np.datetime64('2015-11-20'),
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
	d = pandas.io.parsers.read_csv('course_report_latest-person_course_day_SW12x.csv.gz', compression='gzip')
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
		
def computeCourseDates (courseId):
	T0 = START_DATES[courseId]
	Tc = T0 + np.timedelta64(7, 'D')
	return T0, Tc

# Get users whose start_date is before Tc and who participated in course
def getRelevantUsers (pc, Tc):
	idxs = np.nonzero((pc.start_time < Tc) & (pc.viewed == 1))[0]
	return pc.username.iloc[idxs]

def getXandY (pc, pcd, usernames, T0, Tc):
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

	# DEBUG -- Only test specific features
	pcd = pcd[['nevents']]
	# END DEBUG

	NUM_DAYS = int((Tc - T0) / np.timedelta64(1, 'D'))
	NUM_FEATURES = NUM_DAYS * len(pcd.columns) + len(pc.columns)
	X = np.zeros((len(usernames), NUM_FEATURES))
	y = np.zeros(len(usernames))
	goodIdxs = []
	for i, username in enumerate(usernames):
		idxs = usernamesToPcdIdxsMap[username]
		# For each row in the person-course-day dataset for this user, put the
		# features into the correct column range for that user in the design matrix X.
		for idx in idxs:
			dateIdx = int((np.datetime64(pcdDates.iloc[idx]) - T0) / np.timedelta64(1, 'D'))
			startColIdx = len(pcd.columns) * dateIdx
			X[i,startColIdx:startColIdx+len(pcd.columns)] = pcd.iloc[idx]
		# Now append the demographic features
		demographics = pc.iloc[usernamesToPcIdxsMap[username]]
		#X[i,NUM_DAYS * len(pcd.columns):] = demographics
		y[i] = usernamesToCertifiedMap[username]
		if np.isfinite(np.sum(X[i,:])):
			goodIdxs.append(i)
	return X[goodIdxs,:], y[goodIdxs]

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

def split (X, y):
	idxs = np.random.permutation(X.shape[0])
	numTraining = int(len(idxs) * 0.8)
	trainIdxs = idxs[0:numTraining]
	testIdxs = idxs[numTraining:]
	return X[trainIdxs,:], y[trainIdxs], X[testIdxs,:], y[testIdxs]

if __name__ == "__main__":
	if 'pcd' not in globals():
		pcd = loadPersonCourseDayData()
		pc = loadPersonCourseData()
	for courseId in pcd.keys():  # For each course
		print courseId
		# Restrict analysis to rows of PC dataset relevant to this course
		idxs = np.nonzero(pc.course_id == courseId)[0]
		somePc = pc.iloc[idxs]
		idxs = np.nonzero(pcd[courseId].course_id == courseId)[0]
		somePcd = pcd[courseId].iloc[idxs]

		# Find start date T0 and cutoff date Tc
		T0, Tc = computeCourseDates(courseId)
		print T0
		usernames = getRelevantUsers(somePc, Tc)

		# Get features and target values
		X, y = getXandY(somePc, somePcd, usernames, T0, Tc)
		if len(np.nonzero(y == 0)[0]) < MIN_EXAMPLES or len(np.nonzero(y == 1)[0]) < MIN_EXAMPLES:
			continue

		# Split into training and testing folds
		trainX, trainY, testX, testY = split(X, y)
		trainX, testX = normalize(trainX, testX)

		# Train and validate
		mlr = sklearn.linear_model.LogisticRegression()
		mlr.fit(trainX, trainY)
		yhat = mlr.predict_proba(testX)[:,1]
		auc = sklearn.metrics.roc_auc_score(testY, yhat)
		print "AUC={}".format(auc)
