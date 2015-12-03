import pandas
import common
import numpy as np

def computeNumCourses (allDisciplines, disciplines, idxs):
	numCoursesVec = np.zeros(len(allDisciplines) + 1, dtype=np.int32)
	for i in range(len(allDisciplines)):
		for j in range(len(idxs)):
			if allDisciplines[i] == disciplines.iloc[idxs[j]]:
				numCoursesVec[i] += 1
	numCoursesVec[-1] = np.sum(numCoursesVec[0:-1])
	return list(numCoursesVec)

def getDiscipline (courseId, courseDisciplinesMap):
	idx1 = courseId.index('/')
	idx2 = courseId.index('/', idx1+1)
	shortCourseId = courseId[idx1+1:idx2]
	return courseDisciplinesMap[shortCourseId]

def getEnrollmentsAndDisciplines ():
	courseDisciplines = pandas.io.parsers.read_csv('course_disciplines.csv')
	courseDisciplinesMap = { courseDisciplines.shortCourseId.iloc[i]:courseDisciplines.discipline.iloc[i] for i in range(courseDisciplines.shape[0]) }
	d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_jwaldo/BigQuery/person_course_HarvardX_2015-11-11-051632.csv')

	# Restrict analysis only to course registrations in which the user participated (viewed=1)
	d = d.iloc[np.nonzero(d.viewed == 1)]

	# Append discipline information
	uniqueCourseIds = np.unique(d.course_id)
	courseIdMap = {}
	for i in range(len(uniqueCourseIds)):
		courseIdMap[uniqueCourseIds[i]] = getDiscipline(uniqueCourseIds[i], courseDisciplinesMap)
	disciplines = d.course_id.map(courseIdMap)
	allDisciplines = np.unique(courseIdMap.values())
	return d, disciplines, allDisciplines

def splitAndWrite (d, trainFrac, testFrac, trainFilename, testFilename, holdoutFilename, columns):
	# Divide user_id's into train, test, and holdout sets.
	userIds = np.unique(d.user_id)
	numTrainIdxs = int(trainFrac * len(userIds))
	numTestIdxs = int(testFrac * len(userIds))
	def getSet (i):
		if i < numTrainIdxs:
			return 0
		elif i < numTrainIdxs+numTestIdxs:
			return 1
		else:
			return 2
	idxs = np.random.permutation(len(userIds))
	userIdMap = { userIds[i]:getSet(idxs[i]) for i in range(len(userIds)) }  # Use "idxs" to randomize
	setAssignments = d.user_id.map(userIdMap)

	# Write each subset to disk
	e = pandas.DataFrame(d, columns=columns)
	e.iloc[np.nonzero(setAssignments == 0)[0]].to_csv(trainFilename, index=False)
	e.iloc[np.nonzero(setAssignments == 1)[0]].to_csv(testFilename, index=False)
	e.iloc[np.nonzero(setAssignments == 2)[0]].to_csv(holdoutFilename, index=False)

def convertStartTimes (d):
	goodIdxs = []
	idx = 0
	startDates = []
	for dateStr in d.start_time:
		try:
			date = np.datetime64(dateStr[0:10])  # "0:10" -- only extract the date
			startDates.append(date)
			goodIdxs.append(idx)
		except:
			pass
		idx += 1
	d = d.iloc[goodIdxs]
	startDates = np.array(startDates, dtype=np.datetime64)
	d.start_time = startDates
	return d

def createIndividualEnrollmentDataset ():
	d, disciplines, allDisciplines = getEnrollmentsAndDisciplines()
	d = convertStartTimes(d)
	fields = [ 'course_id', 'explored', 'start_time', 'user_id', 'ndays_act'  ] + common.DEMOGRAPHIC_FIELDS
	d = d[fields]

	# Remove all rows containing any NaNs
	idxs = np.nonzero(np.sum(pandas.isnull(d), axis=1) == 0)[0]
	d = d.iloc[idxs]

	# Remove identifying information by mapping from username to an integer
	userIds = np.unique(d.user_id)
	userIds = userIds[np.random.permutation(len(userIds))]
	userIdsMap = { userIds[i]:i for i in range(len(userIds)) }
	d.user_id = d.user_id.map(userIdsMap)

	splitAndWrite(d, 0.7, 0.2, "train_individual.csv", "test_individual.csv", "holdout_individual.csv", d.columns)

def createAggregateEnrollmentDataset ():
	d, disciplines, allDisciplines = getEnrollmentsAndDisciplines()
	# For now, stick to fields that rarely change among different courses for
	# the same user.
	fields = [ 'user_id', 'username', 'ip', 'cc_by_ip', 'countryLabel', 'continent', 'city', 'region', 'subdivision', 'postalCode', 'un_major_region', 'un_economic_group', 'un_developing_nation', 'un_special_region', 'latitude', 'longitude', 'LoE', 'YoB', 'gender' ]
	d = d[fields]

	# Create a map from user_id to the indices of rows corresponding to that user
	userIdsMap = {}
	for i in range(d.shape[0]):
		if i % 1000 == 0:
			print i
		userId = d.iloc[i].user_id
		userIdsMap.setdefault(userId, [])
		userIdsMap[userId].append(i)

	# Process rows of each unique user
	e = []
	count = 0
	for userId in userIdsMap.keys():
		idxs = userIdsMap[userId]
		numCoursesVec = computeNumCourses(allDisciplines, disciplines, idxs)
		if numCoursesVec[-1] > 0:
			row = list(d.iloc[idxs[0]].as_matrix())
			e.append(numCoursesVec + row)
		count += 1
		if count % 1000 == 0:
			print count

	columns = fields + [ ('numCourses'+allDisciplines[i].replace(' ', '')) for i in range(len(allDisciplines)) ] + [ 'numCoursesAll' ]
	splitAndWrite(e, 0.7, 0.2, "train.csv", "test.csv", "holdout.csv", columns)

if __name__ == "__main__":
	#createAggregateEnrollmentDataset()
	createIndividualEnrollmentDataset()
