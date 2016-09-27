import tarfile
import numpy as np
import datetime
import pandas
import xml.sax.saxutils
import xml.etree.ElementTree as ET
import os
import re
NUM_WEEKS_HEURISTIC = 2
HARVARDX = "HarvardX/"
CHARLESRIVERX_COURSE_ROOT = "/nfs/home/J/jwhitehill/shared_space/ci3_charlesriverx/HarvardX/Courses"
HX_COURSE_ROOT = "/nfs/home/J/jwhitehill/shared_space/ci3_jwaldo/Harvard"
RE = re.compile(r'^.*\/course\/[^\/]*.xml$')

def convertFieldsToDummies (pc):
	continent = [ 'Europe', 'Oceania', 'Africa', 'Asia', 'Americas', 'North America', 'South America' ]
	LoE = [ 'a', 'none', 'b', 'el', 'hs', 'm', 'p', 'jhs', 'other', 'p_se', 'p_oth']
	gender = ['null', 'm', 'o', 'f']
	variableNames = [ 'continent', 'LoE', 'gender' ]
	variables = [ continent, LoE, gender ]
	# Now handle individual values of each field
	for i, variableName in enumerate(variableNames):
		theMap = { variables[i][j]:j+1 for j in range(len(variables[i])) }
		pc[variableName] = pc[variableName].map(theMap, na_action='ignore')
	# Now handle NaN
	pc = pc.fillna({ variableName:0 for variableName in variableNames })
	return pc

def extractDate (xmlStr):
	convertedStr = xml.sax.saxutils.unescape(xmlStr).replace("\"", "")  # Remove quotes
	if convertedStr == "null":
		raise LookupError("Invalid value for date")
	return np.datetime64(convertedStr)

def convertTimes (d, colName):
        goodIdxs = []
        idx = 0
        dates = []
        for dateStr in d[colName]:
                try:
                        date = np.datetime64(dateStr[0:10])  # "0:10" -- only extract the date
                        dates.append(date)
                        goodIdxs.append(idx)
                except:
                        pass
                idx += 1
        d = d.iloc[goodIdxs]
        dates = np.array(dates, dtype=np.datetime64)
        d[colName] = dates
        return d

def loadData (courseId):
	directory = CHARLESRIVERX_COURSE_ROOT + "/" + courseId.replace(HARVARDX, "").replace("/", "-")
	pc = pandas.read_csv(directory + "/" + "person_course.csv.gz")
	pc = convertTimes(pc, "start_time")
	pcs = pandas.read_csv(directory + "/" + "person_course_survey.csv.gz")
	pcd = pandas.read_csv(directory + "/" + "person_course_day.csv.gz")
	pcd = convertTimes(pcd, "date")
	return pc, pcs, pcd

def getCourseStartAndEndDates ():
	courseIds = os.listdir(CHARLESRIVERX_COURSE_ROOT)
	startDates = {}
	endDates = {}
	for courseId in courseIds:
		weeks = os.listdir(HX_COURSE_ROOT + "/" + courseId)
		try:
			# In line below, always use most recent week (hopefully
			# it's more likely to actually contain the tarGz file).
			startDate, endDate = parseStartAndEndDateFromTarGz(HX_COURSE_ROOT + "/" + courseId + "/" + weeks[-1] + "/" + "course.xml.tar.gz")
			courseId = "HarvardX" + "/" + courseId.replace('-', '/')
			print courseId, startDate, endDate
			startDates[courseId] = startDate
			endDates[courseId] = endDate
		except (LookupError, tarfile.ReadError):
			pass
	return startDates, endDates
	#return { 'HarvardX/HLS2X/T12016': np.datetime64('2016-03-18T00:00:00') }, \
	#       { 'HarvardX/HLS2X/T12016': np.datetime64('2016-06-18T00:00:00') }

def parseStartAndEndDateFromXml (f):
	tree = ET.parse(f)
	root = tree.getroot()
	if "end" not in root.keys() or "start" not in root.keys():
		raise LookupError("Could not find \"start\" or \"end\" in XML file")
	return extractDate(root.attrib["start"]), extractDate(root.attrib["end"])

def parseStartAndEndDateFromTarGz (tarGzFilename):
	with tarfile.open(tarGzFilename, mode="r:gz") as tf:
		names = tf.getnames()
		for name in names:
			if RE.match(name):
				f = tf.extractfile(name)
				startDate, endDate = parseStartAndEndDateFromXml(f)
				return startDate, endDate
		raise LookupError("Could not parse start/end dates for {}".format(tarGzFilename))
