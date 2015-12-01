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

def splitAndWrite (e, allDisciplines, trainFrac, testFrac, trainFilename, testFilename, holdoutFilename, columns):
	# Divide into train, test, and holdout
	idxs = np.random.permutation(len(e))
	numTrainIdxs = int(trainFrac * len(idxs))
	numTestIdxs = int(testFrac * len(idxs))
	trainIdxs = idxs[0:numTrainIdxs]
	testIdxs = idxs[numTrainIdxs:numTrainIdxs+numTestIdxs]
	holdoutIdxs = idxs[numTrainIdxs+numTestIdxs:]

	# Write each subset to disk
	f = pandas.DataFrame(e, columns=columns)
	f.iloc[trainIdxs].to_csv(trainFilename, index=False)
	f.iloc[testIdxs].to_csv(testFilename, index=False)
	f.iloc[holdoutIdxs].to_csv(holdoutFilename, index=False)

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
	fields = [ 'course_id', 'certified', 'start_time', 'user_id' ] + common.DEMOGRAPHIC_FIELDS
	d = d[fields]

	# Remove all rows containing any NaNs
	idxs = np.nonzero(np.sum(pandas.isnull(d), axis=1) == 0)[0]
	d = d.iloc[idxs]

	# Construct matrix of who started each course and when, along with whether he/she certified in each course
	uniqueCourseIds = np.unique(d.course_id)
	courseIdsIdxsMap = { uniqueCourseIds[i]:i for i in range(len(uniqueCourseIds)) }

	# Create maps from userId to demographics, and from userId to their indices in a table
	userIdsDemographics = d.drop_duplicates(subset="user_id")
	allUserIds = list(userIdsDemographics.user_id)
	userIdsDemographics = userIdsDemographics[common.DEMOGRAPHIC_FIELDS]
	userIdsDemographics = userIdsDemographics.as_matrix()
	userIdsIdxsMap = { allUserIds[i]:i for i in range(len(allUserIds)) }
	
	# Now create a table of which users started/certified in which courses before/after time T
	e = np.zeros((len(userIdsIdxsMap.keys()), len(uniqueCourseIds) * 3), dtype=np.int32)
	T = np.datetime64('2015-06-30')  # Arbitrary cutoff time
	for i in range(d.shape[0]):
		if i % 1000 == 0:
			print i
		userId = d.iloc[i].user_id
		courseId = d.iloc[i].course_id
		certified = d.iloc[i].certified
		startTime = d.iloc[i].start_time
		courseIdx = courseIdsIdxsMap[courseId]
		e[userIdsIdxsMap[userId]][3*courseIdx+0] = startTime < T
		e[userIdsIdxsMap[userId]][3*courseIdx+1] = startTime >= T
		e[userIdsIdxsMap[userId]][3*courseIdx+2] = certified

	# Now concatenate the two tables
	f = np.concatenate((userIdsDemographics, e), axis=1)

	# Split and write to disk
	columns = list(common.DEMOGRAPHIC_FIELDS)
	for i in range(len(uniqueCourseIds)):
		courseName = uniqueCourseIds[i].replace(" ", "").replace("/", "_")
		columns.append("start_" + courseName + "_beforeT")
		columns.append("start_" + courseName + "_afterT")
		columns.append("certified" + courseName)
	splitAndWrite(f, allDisciplines, 0.7, 0.2, "train_individual_courses.csv", "test_individual_courses.csv", "holdout_individual_courses.csv", columns)

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
	splitAndWrite(e, allDisciplines, 0.7, 0.2, "train.csv", "test.csv", "holdout.csv", columns)

if __name__ == "__main__":
	#createAggregateEnrollmentDataset()
	createIndividualEnrollmentDataset()
