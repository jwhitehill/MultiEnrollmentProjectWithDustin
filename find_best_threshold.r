library(glmnet)
library(doParallel)
registerDoParallel(4)
source("utility.r")
if (! exists("d")) {
	d <- loadData("train.csv");
}

X <- model.matrix(numCoursesAll ~ continent + LoE + ageRange + gender, d)

# Try a bunch of different thresholds
for (threshold in 1:30) {
	# Binarize as numCoursesAll > threshold
	y <- matrix((d[,c("numCoursesAll")] > threshold) * 1)

	results <- cv.glmnet(X, y, nfolds=3, family="binomial", type.measure="auc", parallel=TRUE)
	print("MLR no interactions")
	print(threshold);
	print(results$cvm);
}
