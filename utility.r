loadData <- function (filename) {
	d <- read.csv(filename, header = TRUE)
	d <- d[,c("numCoursesAll", "continent", "LoE", "YoB", "gender")]

	# Remove rows with missing values
	d <- d[complete.cases(d),]
	# Remove rows with "" values
	d <- d[d$gender != "",]
	d <- d[d$continent != "",]
	d <- d[d$LoE != "",]

	d$ageRange <- cut(2012 - d$YoB, c(-Inf, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, Inf))

	return(d);
}
