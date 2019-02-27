// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <math.h>
#include <assert.h>

namespace ogmaneo {
// Compressed sparse row (CSR) format
struct SparseMatrix {
	int _rows, _columns; // Dimensions

	std::vector<float> _nonZeroValues;
	std::vector<int> _rowRanges;
	std::vector<int> _columnIndices;

	// Transpose
	std::vector<int> _nonZeroValueIndices;
	std::vector<int> _columnRanges;
	std::vector<int> _rowIndices;

	// --- Init ---

	SparseMatrix() {}

	// If you don't want to construct immediately
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	) {
		init(rows, columns, nonZeroValues, rowRanges, columnIndices);
	}

	// From a non-compressed sparse matrix
	SparseMatrix(
		int rows,
		int columns,
		const std::vector<float> &data
	) {
		init(rows, columns, data);
	}

	// If you don't want to construct immediately
	void init(
		int rows,
		int columns,
		const std::vector<float> &nonZeroValues,
		const std::vector<int> &rowRanges,
		const std::vector<int> &columnIndices
	);

	// From a non-compressed sparse matrix
	void init(
		int rows,
		int columns,
		const std::vector<float> &data
	);

	// Generate a transpose, must be called after the original has been created
	void initT();

	// --- Dense ---

	float multiply(
		const std::vector<float> &in,
		int row
	);

	float distance(
		const std::vector<float> &in,
		int row
	);

	// Count number of elements in each row
	int counts(
		int row
	);

	float counts(
		const std::vector<float> &in,
		int row
	);

	// --- Transpose ---

	float multiplyT(
		const std::vector<float> &in,
		int column
	);

	float distanceT(
		const std::vector<float> &in,
		int column
	);

	// Count number of elements in each column
	int countsT(
		int column
	);

	float countsT(
		const std::vector<float> &in,
		int column
	);

	// --- One-Hot Vectors Operations ---

	float multiplyOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float multiplyOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	float distanceOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float distanceOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	// --- Delta Rules ---

	void deltas(
		const std::vector<float> &in,
		float delta,
		int row
	);

	void deltasT(
		const std::vector<float> &in,
		float delta,
		int column
	);

	void deltaOHVs(
		const std::vector<int> &nonZeroIndices,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaOHVsT(
		const std::vector<int> &nonZeroIndices,
		float delta,
		int column,
		int oneHotSize
	);

	// --- Normalization ---

	void normalize(
		int row
	);

	void normalizeT(
		int column
	);

	// --- Hebb Rules ---

	void hebb(
		const std::vector<float> &in,
		int row,
		float alpha
	);

	void hebbOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float alpha
	);

	void hebbDecreasing(
		const std::vector<float> &in,
		int row,
		float alpha
	);

	void hebbDecreasingOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float alpha
	);
};
} // namespace ogmaneo