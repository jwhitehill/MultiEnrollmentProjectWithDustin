library(texreg)
source("utility.r")
if (! exists("d")) {
	d <- loadData("train.csv");
}

d$takeMany = d$numCoursesAll > 1  # Did the learner take >1 courses?
model <- glm(takeMany ~ continent + LoE + ageRange + gender, family = "binomial", data = d)
screenreg(model)
plotreg(model)
