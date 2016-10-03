import cPickle
import scipy.stats
import pandas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from common import NUM_WEEKS_HEURISTIC, CHARLESRIVERX_COURSE_ROOT, HARVARDX, convertTimes, loadCourseDates

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
	
def computeOverallMedianAccuracy ():
	resultsCertRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	resultsCertRepeatedCourseDemog = cPickle.load(open("results_prong1_demog.pkl", "rb"))
	resultsCertHeuristic = cPickle.load(open("results_heuristic.pkl", "rb"))
	(resultsNextWeek, usernames, resultsCert) = cPickle.load(open("results_prong2.pkl", "rb"))

	print "Median accuracy for Approach 1: {}".format(np.median(np.hstack(resultsCertRepeatedCourse.values())))
	print "Median accuracy for Approach 2: {}".format(np.median(np.hstack(resultsCert.values())))
	print "Median accuracy for Approach 3: {}".format(np.median(np.hstack(resultsCertHeuristic.values())))

def printAccuracies ():
	(resultsNextWeek, usernames, resultsCert) = cPickle.load(open("results_prong2.pkl", "rb"))

	for courseId in resultsCert.keys():
		someResultsCert = np.array(resultsCert[courseId])
		print courseId, someResultsCert

def myMedian (x):
	x = np.array(x)
	return np.median(x[np.nonzero(np.isfinite(x))[0]])

def myMean (x):
	x = np.array(x)
	return np.mean(x[np.nonzero(np.isfinite(x))[0]])

def reportBroadStats ():
	d = pandas.read_csv('course_to_discipline.csv')
	courseToDiscipline = { d.course_id.iloc[i]:d.discipline_grouping.iloc[i] for i in range(len(d)) }
	START_DATES, PREDICTION_DATES_0_5, PREDICTION_DATES_1_0 = loadCourseDates()
	resultsRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	allStudents = set
	n = 0
	for courseId in resultsRepeatedCourse.keys():
		if len(resultsRepeatedCourse[courseId]) > 0:
			directory = CHARLESRIVERX_COURSE_ROOT + "/" + courseId.replace(HARVARDX, "").replace("/", "-")
			pc = pandas.read_csv(directory + "/" + "person_course.csv.gz")
			pc = convertTimes(pc, "start_time")
			numRegistrants = np.sum(pc.start_time < PREDICTION_DATES_1_0[courseId])
			n += numRegistrants
			idxs = np.nonzero((pc.start_time < PREDICTION_DATES_1_0[courseId]) & (pc.viewed == 1))[0]
			allStudents = allStudents.union(set(pc.username))
			pc = pc.iloc[idxs]
			print "{} & {} & {} & {} & {}\\\\".format( \
			  courseId, courseToDiscipline[courseId], numRegistrants, pc.shape[0], pc.certified.sum() \
			)
	print "Total participants: {}".format(len(allStudents))

def predictAccuracies ():
	resultsRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	d = pandas.read_csv('course_to_discipline.csv')
	e = pandas.read_csv('rcodes_by_course_id_10-03-2016.csv')
	courseToIdxMap = { e.course_id.iloc[i]:i for i in range(len(e)) }
	#courseToDiscipline = { d.course_id.iloc[i]:d.discipline_grouping.iloc[i] for i in range(len(d)) }
	#accuraciesByDiscipline = {}
	#disciplines = set(courseToDiscipline.values()) - set([ 'Alumni' ])
	#for discipline in disciplines:
	#	accuraciesByDiscipline.setdefault(discipline, [])
	#for courseId in resultsRepeatedCourse.keys():
	#	results = resultsRepeatedCourse[courseId]
	#	if len(results) > 0:
	#		accuraciesByDiscipline[courseToDiscipline[courseId]].append(myMedian(results))
	#for discipline in disciplines:
	#	print "{}: {}".format(discipline, np.median(accuraciesByDiscipline[discipline]))
	#print scipy.stats.mstats.kruskalwallis(*(accuraciesByDiscipline.values()))
	dataByCourse = []
	for courseId in resultsRepeatedCourse.keys():
		results = resultsRepeatedCourse[courseId]
		if len(results) > 0:
			if courseId in courseToIdxMap.keys():
				idx = courseToIdxMap[courseId]
				P = e.P.iloc[idx]
				V = e.V.iloc[idx]
				H = e.H.iloc[idx]
				if np.isfinite(P + V + H):
					dataByCourse.append((myMean(results), P, V, H))
				else:
					print "missing {}".format(courseId)
			else:
				print "missing {}".format(courseId)
	dataByCourse = np.array(dataByCourse)
	print np.corrcoef(dataByCourse.T)
	print scipy.stats.pearsonr(dataByCourse[:,0], dataByCourse[:,2])

def plotAggregateAccuracyCurves ():
	def reverse (x):
		return x[-1::-1]

	def aggregate (listOfLists, idx = None):
		MAX_WEEKS = 8
		MIN_DATA = 4
		lengths = np.array([ len(l) for l in listOfLists ])
		maxLength = min(MAX_WEEKS, np.max(lengths))
		vals = []
		weeks = np.arange(maxLength)
		times = []
		for i in weeks:
			idxs = np.nonzero(lengths > i)[0]
			if len(idxs) >= MIN_DATA:
				if idx == None:
					vals.append(myMean([ reverse(listOfLists[j])[i] for j in idxs ]))
				else:
					vals.append(myMean([ reverse(listOfLists[j])[i][idx] for j in idxs ]))
				times.append(weeks[i])
		return -1 * np.array(reverse(times)), np.array(reverse(vals))

	def doPlot ((t, x), color, lineSymbol):
		handle, = plt.plot(t, x*100., color + lineSymbol)
		plt.plot(t, x*100., color + 'o')
		return handle

	resultsRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	resultsRepeatedCourseDemog = cPickle.load(open("results_prong1_demog.pkl", "rb"))
	resultsCrosstrain = cPickle.load(open("results_xtrain_prong1.pkl", "rb"))
	resultsHeuristic = cPickle.load(open("results_heuristic.pkl", "rb"))
	(_, _, resultsNextWeek) = cPickle.load(open("results_prong2.pkl", "rb"))

	# Gather data
	allResultsRepeatedCourse = []
	allResultsRepeatedCourseDemog = []
	allResultsCrosstrain = []
	allResultsHeuristic = []
	allResultsNextWeek = []
	for courseId in resultsRepeatedCourse.keys():
		if courseId in resultsRepeatedCourse.keys():
			allResultsRepeatedCourse.append(resultsRepeatedCourse[courseId])
		if courseId in resultsRepeatedCourseDemog.keys():
			allResultsRepeatedCourseDemog.append(resultsRepeatedCourseDemog[courseId])
		if courseId in resultsCrosstrain.keys():
			allResultsCrosstrain.append(resultsCrosstrain[courseId])
		if courseId in resultsHeuristic.keys():
			allResultsHeuristic.append(resultsHeuristic[courseId])
		if courseId in resultsNextWeek.keys():
			allResultsNextWeek.append(resultsNextWeek[courseId])

	# Average over courses within each week
	plt.clf()
	handles = []
	handles.append(doPlot(aggregate(allResultsRepeatedCourse), 'y', '-'))
	handles.append(doPlot(aggregate(allResultsRepeatedCourseDemog), 'g', '-'))
	handles.append(doPlot(aggregate(allResultsNextWeek), 'k', '--'))
	handles.append(doPlot(aggregate(allResultsCrosstrain, idx=0), 'c', '-.'))
	handles.append(doPlot(aggregate(allResultsCrosstrain, idx=1), 'b', '-.'))
	handles.append(doPlot(aggregate(allResultsHeuristic), 'm', ':'))
	names = [ "Train on previous offering", "Train on previous offering (demog only)", "Train within course", "Train on different course", "Train on mean course", "Baseline heuristic" ]
	filename = "aggregate_graph.pdf".format(courseId.replace("/", "-"))
	pp = PdfPages(filename)
	plt.legend(handles, names, loc="lower center")
	plt.title("Dropout Prediction Accuracy: Comparison across Approaches")
	plt.xlabel("Week #")
	plt.ylabel("Accuracy (AUC %)")
	#xticks = plt.xticks()[0]
	#plt.xticks(xticks, np.abs(xticks).astype(np.int32))
	plt.xlim((-7.5, 0.5))
	plt.ylim((25., 100.))
	plt.savefig(pp, format="pdf")
	pp.close()

def plotAccuracyCurves ():
	resultsCertRepeatedCourse = cPickle.load(open("results_prong1.pkl", "rb"))
	resultsCertRepeatedCourseDemog = cPickle.load(open("results_prong1_demog.pkl", "rb"))
	resultsCertHeuristic = cPickle.load(open("results_heuristic.pkl", "rb"))
	(resultsNextWeek, usernames, resultsCert) = cPickle.load(open("results_prong2.pkl", "rb"))

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
		filename = "{}_graph.pdf".format(courseId.replace("/", "-"))
		pp = PdfPages(filename)
		plt.legend(handles, names, loc="lower center")
		plt.title(courseId)
		plt.xlabel("Week #")
		plt.ylabel("Accuracy (%)")
		plt.xlim((0., 7.))
		plt.ylim((25., 100.))
		plt.savefig(pp, format="pdf")
		#plt.savefig(filename)
		#plt.show()
		pp.close()

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
	#printAccuracies()
	reportBroadStats()
	#predictAccuracies()
	#plotAggregateAccuracyCurves()
	#plotAccuracyCurves()
	#plotEmpiricalDistributions()
	#plotPH231xPredictions()
	#computeOverallMedianAccuracy()
