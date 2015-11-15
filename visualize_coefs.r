library(texreg)
source("utility.r")
if (! exists("d")) {
	d <- loadData("train.csv");
}

disciplines <- c("HealthSciences", "Humanities", "STEM", "SocialSciences", "All")
for (j in 1:length(disciplines)) {
	discipline <- disciplines[j]

	models <- vector(mode="list", length=3)
	comparisons <- c("0_v_MoreThan0", "1_v_MoreThan1", "AtMost4_v_4OrMore")
	names(models) <- comparisons

	d$take0OrMore <- d[paste("numCourses", discipline, sep="")] > 0
	# Subselect those people who take >0 courses in the specified discipline
	e <- d[d$take0OrMore,]
	e$take1OrMore <- e[paste("numCourses", discipline, sep="")] > 1
	e$take4OrMore <- e[paste("numCourses", discipline, sep="")] > 4

	model <- glm(take0OrMore ~ continent + LoE + age + gender, family = "binomial", data = d)
	models[[1]] <- model
	model <- glm(take1OrMore ~ continent + LoE + age + gender, family = "binomial", data = e)
	models[[2]] <- model
	model <- glm(take4OrMore ~ continent + LoE + age + gender, family = "binomial", data = e)
	models[[3]] <- model

	filename <- paste("models_", discipline, ".pdf", sep="")
	plotreg(models, filename, custom.model.names=comparisons, mfrow=FALSE)
}
