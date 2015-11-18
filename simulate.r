library(texreg)
library(Zelig)
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

	# Subselect those people who take >0 courses in the specified discipline
	take0OrMore <- c(d[paste("numCourses", discipline, sep="")] > 0)
	e <- d[take0OrMore,]

	e$take1OrMore <- c(e[paste("numCourses", discipline, sep="")] > 1)
	model <- zelig(take1OrMore ~ continent + LoE + ageRange + gender, model = "logit", data = e)
	x1 <- setx(model, continent = "North America", LoE = "high school", ageRange = "(20,25]", gender = "m")
	p1 <- sim(model, x1)
	
	x2 <- setx(model, continent = "North America", LoE = "high school", ageRange = "(55,60]", gender = "m")
	p2 <- sim(model, x2)

	print(discipline)
	print("For North American male with high school between 20-25 year of age, prob of taking >1 is: ")
	print(p1)
	print("For North American male with high school between 55-60 year of age, prob of taking >1 is: ")
	print(p2)
}
