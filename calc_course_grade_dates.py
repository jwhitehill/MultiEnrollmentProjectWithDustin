import os
import numpy as np
import pandas

DIR = "/nfs/home/J/jwhitehill/shared_space/ci3_jwaldo/dseaton/For_Jake"
FRAC = 1.0

def computeDate (f, frac):
	d = pandas.read_csv(f)
	if d.shape[0] == 0:
		return None
	threshold = d.threshold
	idxs = np.nonzero(d['Csum Pts'] > threshold * frac)
	date = np.min(d.start.iloc[idxs])
	return date

if __name__ == "__main__":
	files = os.listdir(DIR)
	courseDateMap = {}
	for f in files:
		if ".csv" in f:
			idx = f.index("_ideal_grade.csv")
			course = f[0:idx].replace("__", "/").replace("_", ".")
			filename = DIR + "/" + f
			courseDateMap[course] = computeDate(filename, FRAC)
