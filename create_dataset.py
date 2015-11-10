import pandas
import numpy as np

d = pandas.io.parsers.read_csv('/nfs/home/J/jwhitehill/shared_space/ci3_jwaldo/BigQuery/person_course_HarvardX_2015-10-21-123325.csv')

# Restrict analysis only to course registrations in which the user participated (viewed=1)
d = d.iloc[np.nonzero(d.viewed == 1)]

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

# Dump rows for each unique user
e = []
for userId in userIdsMap.keys():
	idxs = userIdsMap[userId]
	numCourses = len(idxs)
	if numCourses > 0:
		row = list(d.iloc[idxs[0]].as_matrix())
		e.append([ numCourses ] + row)

# Divide into train (70%), test (20%), and holdout (10%)
idxs = np.random.permutation(len(e))
TRAIN_FRAC = 0.7
TEST_FRAC = 0.2
numTrainIdxs = int(TRAIN_FRAC * len(idxs))
numTestIdxs = int(TEST_FRAC * len(idxs))
trainIdxs = idxs[0:numTrainIdxs]
testIdxs = idxs[numTrainIdxs:numTrainIdxs+numTestIdxs]
holdoutIdxs = idxs[numTrainIdxs+numTestIdxs:]

# Write each subset to disk
f = pandas.DataFrame(e, columns=(['numCourses'] + fields))
f.iloc[trainIdxs].to_csv('train.csv', index_label=False)
f.iloc[testIdxs].to_csv('test.csv', index_label=False)
f.iloc[holdoutIdxs].to_csv('holdout.csv', index_label=False)
