library(texreg)
source("utility.r")
if (! exists("d")) {
	d <- loadData("train.csv");
}

THRESHOLDS <- c(1, 4)
for (i in 1:length(THRESHOLDS)) {
	threshold <- THRESHOLDS[i]
	disciplines <- c("Alumni", "HealthSciences", "Humanities", "STEM", "SocialSciences", "All")
	models <- vector(mode="list", length=length(disciplines))
	names(models) <- disciplines
	for (j in 1:length(disciplines)) {
		d$takeMany <- d[paste("numCourses", disciplines[j], sep="")] > threshold
		model <- glm(takeMany ~ continent + LoE + ageRange + gender, family = "binomial", data = d)
		models[[j]] <- model
	}

	filename <- paste("models", threshold, ".pdf", sep="")
	plotreg(models, filename, custom.model.names=disciplines, mfrow=FALSE)
}
