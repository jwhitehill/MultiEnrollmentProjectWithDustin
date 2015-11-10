import PoissonRegression
import common
import util
import numpy as np
import pandas

if __name__ == "__main__":
	d = util.loadTrainingSet()
	fields = list(d.columns)
	fields.remove(common.TARGET_VARIABLE)

	y = d[common.TARGET_VARIABLE]
	X = d[fields]
	pr = PoissonRegression.PoissonRegression(C=0., fit_intercept=True, minValue = 1)
	pr.fit(X, y)

	coefficients = { fields[i]:pr.theta[i] for i in range(len(fields)) }
	intercept = pr.theta[-1]

	yhat = pr.predict(X)
	rmsePR = np.mean((yhat - y) ** 2) ** 0.5
	rmseBaseline = np.mean((np.mean(y) - y) ** 2) ** 0.5
	print "PR: {}  Baseline: {}".format(rmsePR, rmseBaseline)
