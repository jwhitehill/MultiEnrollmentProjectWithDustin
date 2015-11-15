loadData <- function (filename) {
	d <- read.csv(filename, header = TRUE)
	d <- d[,c("numCoursesAll", "numCoursesHealthSciences", "numCoursesHumanities", "numCoursesSTEM", "numCoursesSocialSciences", "continent", "LoE", "YoB", "gender")]

	# Remove rows with missing values
	d <- d[complete.cases(d),]
	# Remove rows with "" values
	d <- d[d$gender != "",]
	d <- d[d$gender != "null",]
	d <- d[d$continent != "",]
	d <- d[d$continent != "null",]
	d <- d[d$LoE != "",]
	d <- d[d$LoE != "null",]
	d <- d[d$LoE != "learn",]
	d <- d[d$LoE != "Learn",]

	d$ageRange <- cut(2012 - d$YoB, c(-Inf, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, Inf))
	d$age <- 2012 - d$YoB

	# Rename education variables to make them more readable (code from Dustin)
	d$LoE <- as.character(d$LoE)  # Convert from factor to character type so we can rename
	d$LoE[d$LoE=="none"]<-"no education"
	d$LoE[d$LoE=="el"]<-"elementary school"
	d$LoE[d$LoE=="jhs"]<-"junior high school"
	d$LoE[d$LoE=="hs"]<-"high school"
	d$LoE[d$LoE=="a"]<-"associate"
	d$LoE[d$LoE=="b"]<-"bachelor"
	d$LoE[d$LoE=="m"]<-"master"
	d$LoE[d$LoE=="p"]<-"professional/post-masters"
	d$LoE[d$LoE=="p_oth"]<-"professional/post-masters"
	d$LoE[d$LoE=="p_se"]<-"professional/post-masters"
	d$LoE <- as.factor(d$LoE)

	return(d);
}
