#include "quantify.hpp"
#include <stdlib.h>
#include <map>

int comparator (const void *el1, const void *el2)
{
	float *f1 = (float *) el1;
	float *f2 = (float *) el2;
	if (*f1 < *f2) {
		return -1;
	} else if (*f1 == *f2) {
		return 0;
	} else {
		return 1;
	}
}

int quantify (const int nRows, const int nCols, float *X)
{
	for (int i = 0; i < nCols; i++) {
		// Sort the column
		float *sortedCol = new float[nRows];
		for (int j = 0; j < nRows; j++) {
			sortedCol[j] = X[j*nCols + i];
		}
		qsort(sortedCol, nRows, sizeof(float), comparator);
		std::map<float, int> m;
		for (int j = 0; j < nRows; j++) {
			m[sortedCol[j]] = j;
		}
	
		// Compute quantile of each element	
		for (int j = 0; j < nRows; j++) {
			float num = X[j*nCols + i];
			float quantile = m[num] / (float) nRows;
			// Replace value with quantile
			X[j*nCols + i] = quantile;
		}

		// Free memory
		delete[] sortedCol;
	}
	return 0;
}
