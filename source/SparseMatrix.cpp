#include "SparseMatrix.h"

using namespace ogmaneo;

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &nonZeroValues,
	const std::vector<int> &rowRanges,
	const std::vector<int> &columnIndices
) {
	_rows = rows;
	_columns = columns;

	_nonZeroValues = nonZeroValues;
	_rowRanges = rowRanges;
	_columnIndices = columnIndices;
}

void SparseMatrix::init(
	int rows,
	int columns,
	const std::vector<float> &data
) {
	_rows = rows;
	_columns = columns;

	_rowRanges.reserve(_rows + 1);
	_rowRanges.push_back(0);

	int nonZeroCountInRow = 0; // Only need to set this to zero once because it's cumulative
	
	for (int row = 0; row < _rows; row++) {
		int rowOffset = row * _columns;

		for (int col = 0; col < _columns; col++) {
			int index = rowOffset + col;

			if (data[index] != 0.0f) {
				_nonZeroValues.push_back(data[index]);
				_columnIndices.push_back(col);

				nonZeroCountInRow++;
			}
		}

		_rowRanges.push_back(nonZeroCountInRow);
	}
}

void SparseMatrix::initT() {
	_columnRanges.resize(_columns + 1, 0);

	_rowIndices.resize(_nonZeroValues.size());

	_nonZeroValueIndices.resize(_nonZeroValues.size());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++)
			_columnRanges[_columnIndices[j]]++;
	}

	// Bring row range array in place using exclusive scan
	int offset = 0;

	for (int i = 0; i < _columns; i++) {
		int temp = _columnRanges[i];

		_columnRanges[i] = offset;

		offset += temp;
	}

	_columnRanges[_columns] = offset;

	std::vector<int> columnOffsets = _columnRanges;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = _rowRanges[i]; j < _rowRanges[nextIndex]; j++) {
			int colIndex = _columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			_rowIndices[nonZeroIndexT] = i;

			_nonZeroValueIndices[nonZeroIndexT] = j;

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
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		sum += _nonZeroValues[j] * in[_columnIndices[j]];

	return sum;
}

int SparseMatrix::counts(
	int row
) {
	int nextIndex = row + 1;
	
	return _rowRanges[nextIndex] - _rowRanges[row];
}

float SparseMatrix::multiplyT(
	const std::vector<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = _columnRanges[column]; j < _columnRanges[nextIndex]; j++)
		sum += _nonZeroValues[_nonZeroValueIndices[j]] * in[_rowIndices[j]];

	return sum;
}

int SparseMatrix::countsT(
	int column
) {
	int nextIndex = column + 1;
	
	return _columnRanges[nextIndex] - _columnRanges[column];
}

float SparseMatrix::multiplyOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = _rowRanges[row]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

		sum += _nonZeroValues[j];
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
	
	for (int jj = _columnRanges[column]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

		sum += _nonZeroValues[_nonZeroValueIndices[j]];
	}

	return sum;
}

void SparseMatrix::deltaOHVs(
	const std::vector<int> &nonZeroIndices,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = _rowRanges[row]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[_columnIndices[jj] / oneHotSize];

		_nonZeroValues[j] += delta;
	}
}

void SparseMatrix::deltaOHVsT(
	const std::vector<int> &nonZeroIndices,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = _columnRanges[column]; jj < _columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[_rowIndices[jj] / oneHotSize];

		_nonZeroValues[_nonZeroValueIndices[j]] += delta;
	}
}

void SparseMatrix::hebb(
	const std::vector<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] += alpha * (in[_columnIndices[j]] - _nonZeroValues[j]);
}

void SparseMatrix::hebbOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int jj = _rowRanges[row]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[_columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			_nonZeroValues[j] += alpha * (target - _nonZeroValues[j]);
		}
	}
}

void SparseMatrix::hebbDecreasing(
	const std::vector<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = _rowRanges[row]; j < _rowRanges[nextIndex]; j++)
		_nonZeroValues[j] += alpha * std::min(0.0f, in[_columnIndices[j]] - _nonZeroValues[j]);
}

void SparseMatrix::hebbDecreasingOHVs(
	const std::vector<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int jj = _rowRanges[row]; jj < _rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[_columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float delta = (dj == targetDJ ? 0.0f : -alpha);

			_nonZeroValues[j] = std::max(0.0f, _nonZeroValues[j] + delta);
		}
	}
}