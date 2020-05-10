// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

using namespace ogmaneo;

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	rows = rows;
	columns = columns;

	this->nonZeroValues = nonZeroValues;
	this->rowRanges = rowRanges;
	this->columnIndices = columnIndices;
}

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &data
) {
	rows = rows;
	columns = columns;

	rowRanges.reserve(rows + 1);
	rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	
	for (int row = 0; row < rows; row++) {
		int rowOffset = row * columns;

		for (int col = 0; col < columns; col++) {
			int index = rowOffset + col;

			if (data[index] != 0.0f) {
				nonZeroValues.push_back(data[index]);
				columnIndices.push_back(col);

				nonZeroCountInRow++;
			}
		}

		rowRanges.push_back(nonZeroCountInRow);
	}
}

void SparseMatrix::initT() {
	columnRanges.resize(columns + 1, 0);

	rowIndices.resize(nonZeroValues.size());

	nonZeroValueIndices.resize(nonZeroValues.size());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++)
			columnRanges[columnIndices[j]]++;
	}

	// Bring row range array in place using exclusive scan
	int offset = 0;

	for (int i = 0; i < columns; i++) {
		int temp = columnRanges[i];

		columnRanges[i] = offset;

		offset += temp;
	}

	columnRanges[columns] = offset;

	std::vector<int> columnOffsets = columnRanges;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++) {
			int colIndex = columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			rowIndices[nonZeroIndexT] = i;

			nonZeroValueIndices[nonZeroIndexT] = j;

			columnOffsets[colIndex]++;
		}
	}
}

float SparseMatrix::multiply(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * in[columnIndices[j]];

	return sum;
}

float SparseMatrix::distance2(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++) {
		float delta = in[columnIndices[j]] - nonZeroValues[j];

		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::count(
	int row
) {
	int nextIndex = row + 1;
	
	return rowRanges[nextIndex] - rowRanges[row];
}

float SparseMatrix::count(
	const std::vector<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += in[columnIndices[j]];

	return sum;
}

void SparseMatrix::fill(
	int row,
    float value
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] = value;
}

float SparseMatrix::total(
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j];

	return sum;
}

float SparseMatrix::multiplyT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * in[rowIndices[j]];

	return sum;
}

float SparseMatrix::distance2T(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++) {
		float delta = in[rowIndices[j]] - nonZeroValues[nonZeroValueIndices[j]];
	
		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::countT(
	int column
) {
	int nextIndex = column + 1;
	
	return columnRanges[nextIndex] - columnRanges[column];
}

float SparseMatrix::countT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += in[rowIndices[j]];

	return sum;
}

void SparseMatrix::fillT(
	int column,
    float value
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] = value;
}

float SparseMatrix::totalT(
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]];

	return sum;
}

float SparseMatrix::multiplyOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		sum += nonZeroValues[j];
	}

	return sum;
}

float SparseMatrix::multiplyOHVsT(
	const std::vector<int> &nonZeroIndices,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		sum += nonZeroValues[nonZeroValueIndices[j]];
	}

	return sum;
}

float SparseMatrix::multiplyOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &nonZeroScalars,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		sum += nonZeroValues[j] * nonZeroScalars[i];
	}

	return sum;
}

float SparseMatrix::multiplyOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &nonZeroScalars,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		sum += nonZeroValues[nonZeroValueIndices[j]] * nonZeroScalars[i];
	}

	return sum;
}

float SparseMatrix::distance2OHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize
) {
	float dist = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			float delta = (dj == targetDJ ? 1.0f : 0.0f) - nonZeroValues[jj + dj];

			dist += delta * delta;
		}
	}

	return dist;
}

float SparseMatrix::distance2OHVsT(
	const std::vector<int> &nonZeroIndices,
	int column,
	int oneHotSize
) {
	float dist = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[rowIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			float delta = (dj == targetDJ ? 1.0f : 0.0f) - nonZeroValues[nonZeroValueIndices[jj + dj]];

			dist += delta * delta;
		}
	}

	return dist;
}

int SparseMatrix::countChangedOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	int row,
	int oneHotSize
) {
	int count = 0;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i])
			count++;
	}

	return count;
}

int SparseMatrix::countChangedOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	int column,
	int oneHotSize
) {
	int count = 0;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		
		if (nonZeroIndices[i] != nonZeroIndicesPrev[i])
			count++;
	}

	return count;
}

float SparseMatrix::multiplyChangedOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			sum += nonZeroValues[j];
		}
	}

	return sum;
}

float SparseMatrix::multiplyChangedOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			sum += nonZeroValues[nonZeroValueIndices[j]];
		}
	}

	return sum;
}

void SparseMatrix::deltas(
	const std::vector<float> &in,
	float delta,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += delta * in[columnIndices[j]];
}

void SparseMatrix::deltasT(
	const std::vector<float> &in,
	float delta,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += delta * in[rowIndices[j]];
}

void SparseMatrix::deltaOHVs(
	const std::vector<int> &nonZeroIndices,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		nonZeroValues[j] += delta;
	}
}

void SparseMatrix::deltaOHVsT(
	const std::vector<int> &nonZeroIndices,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		nonZeroValues[nonZeroValueIndices[j]] += delta;
	}
}

void SparseMatrix::deltaOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &nonZeroScalars,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		nonZeroValues[j] += delta * nonZeroScalars[i];
	}
}

void SparseMatrix::deltaOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<float> &nonZeroScalars,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		nonZeroValues[nonZeroValueIndices[j]] += delta * nonZeroScalars[i];
	}
}

void SparseMatrix::deltaChangedOHVs(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[j] += delta;
		}
	}
}

void SparseMatrix::deltaChangedOHVsT(
	const std::vector<int> &nonZeroIndices,
	const std::vector<int> &nonZeroIndicesPrev,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[nonZeroValueIndices[j]] += delta;
		}
	}
}

void SparseMatrix::hebb(
	const std::vector<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += alpha * (in[columnIndices[j]] - nonZeroValues[j]);
}

void SparseMatrix::hebbT(
	const std::vector<float> &in,
	int column,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += alpha * (in[rowIndices[j]] - nonZeroValues[nonZeroValueIndices[j]]);
}

void SparseMatrix::hebbOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			nonZeroValues[j] += alpha * (target - nonZeroValues[j]);
		}
	}
}

void SparseMatrix::hebbOHVsT(
	const std::vector<int> &nonZeroIndices,
	int column,
	int oneHotSize,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[rowIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			nonZeroValues[nonZeroValueIndices[j]] += alpha * (target - nonZeroValues[nonZeroValueIndices[j]]);
		}
	}
}