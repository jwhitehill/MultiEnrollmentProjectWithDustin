import pandas
import sklearn
import numpy as np
import common

def makeLabels (y):
	labels = np.hstack((1 - np.atleast_2d(y).T, np.atleast_2d(y).T)).astype(np.float64)
	return labels

def showProgressAll (cross_entropy, x, y, y_, test_x, test_y, names):
	ll = cross_entropy.eval({x: test_x, y_: test_y})
	print "LL = {}".format(ll)
	for i, name in enumerate(names):
		auc = sklearn.metrics.roc_auc_score(test_y[:,i], y.eval({x: test_x})[:,i])
		print "AUC ({}) = {}".format(name, auc)

def showProgress (cross_entropy, x, y, y_, test_x, test_y, keep_prob = None):
	if keep_prob == None:
		ll = cross_entropy.eval({x: test_x, y_: test_y})
		auc = sklearn.metrics.roc_auc_score(test_y[:,1], y.eval({x: test_x})[:,1])
	else:
		ll = cross_entropy.eval({x: test_x, y_: test_y, keep_prob: 1.})
		auc = sklearn.metrics.roc_auc_score(test_y[:,1], y.eval({x: test_x, keep_prob: 1.})[:,1])
	print "LL={} AUC={}".format(ll, auc)

def getNonNullData (d):
	# Restrict to rows that are not null
	idxs = np.nonzero(np.sum(d[common.FIELDS].isnull(), axis=1) == 0)[0]
	# Restrict to rows that do not contain the string 'null'
	someFields = list(common.FIELDS)
	someFields.remove('YoB')  # Remove non-string fields
	idxs = idxs[np.nonzero(np.sum(d[someFields].iloc[idxs] == 'null', axis=1) == 0)[0]]
	d = d.iloc[idxs]
	idxs = np.nonzero((d.LoE != 'learn') & (d.LoE != 'Learn'))[0]
	return d.iloc[idxs]

#def createIndicatorVariableMaps ():
#	# Create dictionary mapping from field name to another map for each field of interest
#	ivMaps = {}
#
#	d = pandas.io.parsers.read_csv('train22Oct2015.csv')
#	d.append(pandas.io.parsers.read_csv('test22Oct2015.csv'))
#	d.append(pandas.io.parsers.read_csv('holdout22Oct2015.csv'))
#	d = getNonNullData(d)
#	for field in [ 'countryLabel', 'continent', 'LoE', 'gender' ]:
#		uniqueVals = list(set(d[field]))
#		ivMaps[field] = { uniqueVals[i]:i for i in range(len(uniqueVals)) }
#	
#	return ivMaps

def loadDataset (filename):
	d = pandas.io.parsers.read_csv(filename)
	d = d[common.FIELDS + [ common.TARGET_VARIABLE ]]
	d = getNonNullData(d)

	return pandas.get_dummies(d, columns = [ 'continent', 'LoE', 'gender' ])

if __name__ == "__main__":
	d = loadDataset("train.csv")
