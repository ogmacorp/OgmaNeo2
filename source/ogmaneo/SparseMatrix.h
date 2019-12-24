// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

#include <fstream>

namespace ogmaneo {
class SparseMatrix {
public:
	int _rows, _columns; // Dimensions
	int _numNonZeroValues;
	bool _hasT;

	cl::Buffer _nonZeroValues;
	cl::Buffer _rowRanges;
	cl::Buffer _columnIndices;

	// Transpose
	cl::Buffer _nonZeroValueIndices;
	cl::Buffer _columnRanges;
	cl::Buffer _rowIndices;

	// --- Init ---

	SparseMatrix()
	:
	_rows(0),
	_columns(0),
	_numNonZeroValues(0),
	_hasT(false)
	{}

	void initLocalRF(
		ComputeSystem &cs,
		const Int3 &inSize,
		const Int3 &outSize,
		int radius,
		float valueLower,
		float valueUpper,
		std::mt19937 &rng
	);

	// Generate a transpose, must be called after the original has been created
	void initT(
		ComputeSystem &cs
	);

	void writeToStream(
		ComputeSystem &cs,
		std::ostream &os
	);

	void readFromStream(
		ComputeSystem &cs,
		std::istream &is
	);
};
} // namespace ogmaneo