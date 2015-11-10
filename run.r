library(glmnet)
library(AUC)
source("utility.r")
if (! exists("d")) {
	d <- loadData("train.csv");
}

#approach <- "MLR"
approach <- "PR"

if (approach == "PR") {
	y <- matrix(d[,c("numCoursesAll")])

	# No interactions
	X <- model.matrix(numCoursesAll ~ continent + LoE + ageRange + gender, d)
	model <- glmnet(X, y, family="poisson", lambda=0)
	yhat = predict(model, X, type="response")
	print("PR no interactions")
	cor(yhat[,ncol(yhat)], y)

	# All 2-way interactions
	X <- model.matrix(numCoursesAll ~ (continent + LoE + ageRange + gender)^2, d)
	model <- glmnet(X, y, family="poisson", lambda=0)
	yhat2 = predict(model, X, type="response")
	print("PR with 2-way interactions")
	cor(yhat2[,ncol(yhat2)], y)
} else if (approach == "MLR") {
	# No interactions
	X <- model.matrix(numCoursesAll ~ continent + LoE + ageRange + gender, d)

	THRESHOLD <- 1
	y <- matrix((d[,c("numCoursesAll")] > THRESHOLD) * 1)  # Binarize as numCoursesAll > THRESHOLD

	model <- glmnet(X, y, family="binomial", lambda=0)
	yhat = predict(model, X, type="response")
	yhat = data.frame(yhat)[,2]
	print("MLR no interactions")
	auc2 = auc(roc(yhat, factor(y)))
	print(auc2);

	# All 2-way interactions
	X <- model.matrix(numCoursesAll ~ (continent + LoE + ageRange + gender)^2, d)
	model <- glmnet(X, y, family="binomial", lambda=0)
	yhat2 = predict(model, X, type="response")
	yhat2 = data.frame(yhat2)[,2]
	print("MLR with 2-way interactions")
	auc(roc(yhat2, factor(y)))
}
