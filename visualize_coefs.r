library(texreg)
source("utility.r")
if (! exists("d")) {
	d <- loadData("all.csv");
}

disciplines <- c("All", "HealthSciences", "Humanities", "STEM", "SocialSciences")
allModels <- vector(mode="list", length=5)
comparisons <- c("0_v_MoreThan0", "1_v_MoreThan1")
for (j in 1:length(disciplines)) {
	discipline <- disciplines[j]

	models <- vector(mode="list", length=2)
	names(models) <- comparisons

	d$take0OrMore <- c(d[paste("numCourses", discipline, sep="")] > 0)
	# Subselect those people who take >0 courses in the specified discipline
	e <- d[d$take0OrMore,]
	e$take1OrMore <- c(e[paste("numCourses", discipline, sep="")] > 1)

	model <- glm(take0OrMore ~ continent + LoE + ageRange + gender, family = "binomial", data = d)
	models[[1]] <- model

	model <- glm(take1OrMore ~ continent + LoE + ageRange + gender, family = "binomial", data = e)
	models[[2]] <- model

	filename <- paste("models_", discipline, ".pdf", sep="")
	# plotreg(models, filename, custom.model.names=comparisons, mfrow=FALSE)
	allModels[[j]] <- models
}
plotreg(allModels[[1]][[2]], custom.model.names=comparisons[2], mfrow=FALSE)

# Histograms
numCourses = c(1:10)
colors = c(rgb(1,0,0,1/4), rgb(0,1,0,1/4), rgb(0,0,1,1/4), rgb(0.5,0.5,0,1/4), rgb(0.5,0,0.5,1/4))
allHists <- vector(mode="list", length=5)
allTitles <- vector(mode="list", length=5)
for (j in 1:length(disciplines)) {
  discipline <- disciplines[j]
  fieldName <- paste("numCourses", discipline, sep="")
  allTitles[[j]] <- paste("Number of courses (", discipline, ")", sep="")
  
  if (j != 1) {  # 1="All"; restrict to students who took >0 courses in this discipline
    d$take0OrMore <- c(d[fieldName] > 0)
    # Subselect those people who take >0 courses in the specified discipline
    e <- d[d$take0OrMore,]
  } else {
    e <- d
  }
  allHists[[j]] <- hist(e[e[fieldName] < max(numCourses),fieldName], right=FALSE, breaks=numCourses, plot=FALSE)
}
par(mfrow=c(1,2))
plot(allHists[[j]], col=colors[[j]], xaxt="n")
axis(1, at=numCourses+0.5, labels=numCourses)
mytable <- table(e[e[fieldName] < max(numCourses),fieldName])
mytable <- mytable[1:4]
labs <- paste(names(mytable), " (",signif(100*mytable/sum(mytable), digits=2),"%)", sep="")
pie(mytable, labels=labs, cex=0.6, radius=1.0, main="% of students\ntaking multiple courses")