import pandas
import numpy as np
import common

def getNonNullData (d):
	# Restrict to rows that are not null
	idxs = np.nonzero(np.sum(d[common.FIELDS].isnull(), axis=1) == 0)[0]
	# Restrict to rows that do not contain the string 'null'
	someFields = list(common.FIELDS)
	someFields.remove('YoB')  # Remove non-string fields
	idxs = idxs[np.nonzero(np.sum(d[someFields].iloc[idxs] == 'null', axis=1) == 0)[0]]
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

def loadTrainingSet ():
	d = pandas.io.parsers.read_csv('train22Oct2015.csv')
	d = d[common.FIELDS + [ common.TARGET_VARIABLE ]]
	d = getNonNullData(d)

	return pandas.get_dummies(d, columns = [ 'countryLabel', 'continent', 'LoE', 'gender' ])

if __name__ == "__main__":
	d = loadTrainingSet()
