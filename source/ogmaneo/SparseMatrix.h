// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
	int rows, columns; // Dimensions

	std::vector<float> nonZeroValues;
	std::vector<int> rowRanges;
	std::vector<int> columnIndices;

	// Transpose
	std::vector<int> nonZeroValueIndices;
	std::vector<int> columnRanges;
	std::vector<int> rowIndices;

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

	float distance2(
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

	float multiplyT(
		const std::vector<float> &in,
		int column
	);

	float distance2T(
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

	float distance2OHVs(
		const std::vector<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float distance2OHVsT(
		const std::vector<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	int countChangedOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
		int row,
		int oneHotSize
	);

	int countChangedOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
		int column,
		int oneHotSize
	);

	float multiplyChangedOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
		int row,
		int oneHotSize
	);

	float multiplyChangedOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
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

	void deltaChangedOHVs(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaChangedOHVsT(
		const std::vector<int> &nonZeroIndices,
		const std::vector<int> &nonZeroIndicesPrev,
		float delta,
		int column,
		int oneHotSize
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
};
} // namespace ogmaneo