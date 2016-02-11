import cPickle
import matplotlib.pyplot as plt
import numpy as np
from predict_certification import NUM_WEEKS_HEURISTIC

resultsCertRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
resultsCertHeuristic = cPickle.load(open("results_heuristic.pkl", "rb"))
(resultsNextWeek, resultsCert) = cPickle.load(open("results_prong2.pkl", "rb"))

for courseId in resultsCertRepeatedCourse.keys():
	plt.clf()

	someResultsCertRepeatedCourse = np.array(resultsCertRepeatedCourse[courseId])
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
		names.append("Certify (trained from prev. course)")
	if len(someResultsCertHeuristic) > 0:
		handle, = plt.plot(np.arange(len(someResultsCertHeuristic)) + NUM_WEEKS_HEURISTIC, someResultsCertHeuristic*100., 'c-')
		handles.append(handle)
		plt.plot(np.arange(len(someResultsCertHeuristic)) + NUM_WEEKS_HEURISTIC, someResultsCertHeuristic*100., 'co')
		names.append("Certify (simple classifier)")
	if len(someResultsCert) > 0:
		handle, = plt.plot(np.arange(len(someResultsCert)) + 3, someResultsCert*100., 'm-')
		handles.append(handle)
		plt.plot(np.arange(len(someResultsCert)) + 3, someResultsCert*100., 'mo')
		names.append("Certify (trained within course)")
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
