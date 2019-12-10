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

	int count(
		int row
	);

	float count(
		const std::vector<float> &in,
		int row
	);

    void fill(
        int row,
        float value
    );

    float total(
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

	int countT(
		int column
	);

	float countT(
		const std::vector<float> &in,
		int column
	);

    void fillT(
        int column,
        float value
    );

    float totalT(
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

	float multiplyOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		int row,
		int oneHotSize
	);

	float multiplyOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		int column,
		int oneHotSize
	);

	float minOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float minOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	float maxOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float maxOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	float countsOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &in,
		int row,
		int oneHotSize
	);

	float countsOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &in,
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

	void deltaOHVs(
		const std::vector<int> &nonZeroIndices,
		float delta,
		int row,
		int oneHotSize,
		float lowerBound,
		float upperBound
	);

	void deltaOHVsT(
		const std::vector<int> &nonZeroIndices,
		float delta,
		int column,
		int oneHotSize,
		float lowerBound,
		float upperBound
	);

	void deltaOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		float delta,
		int column,
		int oneHotSize
	);

	void deltaOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		float delta,
		int row,
		int oneHotSize,
		float lowerBound,
		float upperBound
	);

	void deltaOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<float> &nonZeroScalars,
		float delta,
		int column,
		int oneHotSize,
		float lowerBound,
		float upperBound
	);

	// --- Normalization ---

	void normalize(
		int row
	);

	void normalizeT(
		int column
	);

	float magnitude2(
		int row
	);

	float magnitude2T(
		int column
	);

	// --- Copy ---

	void copyRow(
		const SparseMatrix &source,
		int row
	);

	void copyColumn(
		const SparseMatrix &source,
		int column
	);

	// --- Hebb Rules ---

	void hebb(
		const std::vector<float> &in,
		int row,
		float alpha
	);

	void hebbT(
		const std::vector<float> &in,
		int column,
		float alpha
	);

	void hebbDecreasing(
		const std::vector<float> &in,
		int row,
		float alpha
	);

	void hebbDecreasingT(
		const std::vector<float> &in,
		int column,
		float alpha
	);

	void hebbOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float alpha
	);

	void hebbOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize,
		float alpha
	);

	void hebbErrors(
		const std::vector<float> &errors,
		int row
	);

	void hebbErrorsT(
		const std::vector<float> &errors,
		int column
	);

    void hebbDecreasingOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float alpha
	);

    void hebbDecreasingOHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize,
		float alpha
	);

	// --- Special ---
	
	float multiplyNoDiagonalOHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);
};
} // namespace ogmaneo