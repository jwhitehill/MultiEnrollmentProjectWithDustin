import cPickle
import matplotlib.pyplot as plt
import numpy as np
from predict_certification import NUM_WEEKS_HEURISTIC

def plotEmpiricalDistributions ():
	allDists = cPickle.load(open("results_prong1_dists.pkl", "rb"))
	for courseId in [ 'HarvardX/SW25x/1T2014', 'HarvardX/SW12.10x/1T2015' ]:
		for weekIdx in range(4):
			y = allDists[courseId][weekIdx][0]
			yhat = allDists[courseId][weekIdx][1]
			bins = np.arange(0, 1.1, 0.1)
			plt.clf()
			plt.hist(y, bins, label='True probability', color='k')
			plt.hist(yhat, bins, label='Estimated probability', color='b', hatch='/')
			plt.legend(loc='upper center')
			plt.xlabel("Probability of certification")
			plt.ylabel("Number of students")
			plt.title("{} week {}".format(courseId, weekIdx+1))
			plt.savefig("{}_week_{}.png".format(courseId.replace("/", "-"), weekIdx+1))
			#plt.show()

def plotAccuracyCurves ():
	resultsCertRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	resultsCertRepeatedCourseDemog = cPickle.load(open("results_prong1_demog.pkl", "rb"))
	resultsCertHeuristic = cPickle.load(open("results_heuristic.pkl", "rb"))
	(resultsNextWeek, resultsCert) = cPickle.load(open("results_prong2.pkl", "rb"))

	for courseId in resultsCertRepeatedCourse.keys():
		plt.clf()

		someResultsCertRepeatedCourse = np.array(resultsCertRepeatedCourse[courseId])
		someResultsCertRepeatedCourseDemog = np.array(resultsCertRepeatedCourseDemog[courseId])
		someResultsCertHeuristic = np.array(resultsCertHeuristic[courseId])
		someResultsNextWeek = np.array(resultsNextWeek[courseId])
		someResultsCert = np.array(resultsCert[courseId])

		# Results from prong #1 are from weeks 1 and later; hence, "+ 1"
		# Results from prong #2 are from weeks 3 and later; hence, "+ 3"
		names = []
		handles = []
		if len(someResultsCertRepeatedCourse) > 0:
			handle, = plt.plot(np.arange(len(someResultsCertRepeatedCourse)) + 1, someResultsCertRepeatedCourse*100., 'y-')
			handles.append(handle)
			plt.plot(np.arange(len(someResultsCertRepeatedCourse)) + 1, someResultsCertRepeatedCourse*100., 'yo')
			names.append("Certify (using prev. course)")
		#if len(someResultsCertRepeatedCourseDemog) > 0:
		#	handle, = plt.plot(np.arange(len(someResultsCertRepeatedCourseDemog)) + 1, someResultsCertRepeatedCourseDemog*100., 'r-')
		#	handles.append(handle)
		#	plt.plot(np.arange(len(someResultsCertRepeatedCourseDemog)) + 1, someResultsCertRepeatedCourseDemog*100., 'ro')
		#	names.append("Certify (demog. only; using prev. course)")
		if len(someResultsCertHeuristic) > 0:
			handle, = plt.plot(np.arange(len(someResultsCertHeuristic)) + NUM_WEEKS_HEURISTIC, someResultsCertHeuristic*100., 'c-')
			handles.append(handle)
			plt.plot(np.arange(len(someResultsCertHeuristic)) + NUM_WEEKS_HEURISTIC, someResultsCertHeuristic*100., 'co')
			names.append("Certify (simple classifier)")
		if len(someResultsCert) > 0:
			handle, = plt.plot(np.arange(len(someResultsCert)) + 3, someResultsCert*100., 'm-')
			handles.append(handle)
			plt.plot(np.arange(len(someResultsCert)) + 3, someResultsCert*100., 'mo')
			names.append("Certify (train within course)")
		if len(someResultsNextWeek) > 0:
			handle, = plt.plot(np.arange(len(someResultsNextWeek)) + 3, someResultsNextWeek*100., 'k-.')
			handles.append(handle)
			plt.plot(np.arange(len(someResultsNextWeek)) + 3, someResultsNextWeek*100., 'ko')
			names.append("Persist to next week")
		plt.legend(handles, names, loc="lower center")
		plt.title(courseId)
		plt.xlabel("Week #")
		plt.ylabel("Accuracy (%)")
		plt.xlim((0., 7.))
		plt.ylim((25., 100.))
		plt.savefig("{}_graph.png".format(courseId.replace("/", "-")))
		#plt.show()

def plotPH231xPredictions ():
	results = cPickle.load(open('results_prong2.pkl', 'rb'))
	allDists = results[2]
	COURSE_ID = 'HarvardX/PH231x/1T2016'
	dists = allDists[COURSE_ID]
	for weekIdx in range(len(dists)):
		yhat = dists[weekIdx][1]
		bins = np.arange(0, 1.1, 0.1)

		plt.clf()
		plt.hist(yhat, bins, label='Estimated probability', color='b', hatch='/')
		plt.legend(loc='upper right')
		plt.xlabel("Probability of certification")
		plt.ylabel("Number of students")
		plt.title("{} week {}".format(COURSE_ID, weekIdx+3))
		plt.savefig("{}_week_{}_predictions.png".format(COURSE_ID.replace("/", "-"), weekIdx+1))

if __name__ == "__main__":
	#plotAccuracyCurves()
	#plotEmpiricalDistributions()
	plotPH231xPredictions()
