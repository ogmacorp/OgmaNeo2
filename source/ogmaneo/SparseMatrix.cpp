// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

using namespace ogmaneo;

void SparseMatrix::initLocalRF(
	ComputeSystem &cs,
    const Int3 &inSize,
    const Int3 &outSize,
    int radius,
	float valueLower,
	float valueUpper,
	std::mt19937 &rng
) {
    int numOut = outSize.x * outSize.y * outSize.z;

    // Projection constant
    Float2 outToIn = Float2(static_cast<float>(inSize.x) / static_cast<float>(outSize.x),
        static_cast<float>(inSize.y) / static_cast<float>(outSize.y));

    int diam = radius * 2 + 1;

    int numWeightsPerOutput = diam * diam * inSize.z;

    int weightsSize = numOut * numWeightsPerOutput;

    std::vector<cl_float> nonZeroValues;
	nonZeroValues.reserve(weightsSize);

    std::vector<cl_int> rowRanges(numOut + 1);

    std::vector<cl_int> columnIndices;
	columnIndices.reserve(weightsSize);

	std::uniform_real_distribution<float> valueDist(valueLower, valueUpper);

    // Initialize weight matrix
    for (int ox = 0; ox < outSize.x; ox++)
        for (int oy = 0; oy < outSize.y; oy++) {
            Int2 visiblePositionCenter = project(Int2(ox, oy), outToIn);

            // Lower corner
            Int2 fieldLowerBound(visiblePositionCenter.x - radius, visiblePositionCenter.y - radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
            Int2 iterUpperBound(std::min(inSize.x - 1, visiblePositionCenter.x + radius), std::min(inSize.y - 1, visiblePositionCenter.y + radius));

            for (int oz = 0; oz < outSize.z; oz++) {
                Int3 outPos(ox, oy, oz);

                int nonZeroInRow = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        for (int iz = 0; iz < inSize.z; iz++) {
                            Int3 inPos(ix, iy, iz);

                            int inIndex = address3(inPos, inSize);

                            nonZeroValues.push_back(valueDist(rng));
                            columnIndices.push_back(inIndex);
                            
                            nonZeroInRow++;
                        }
                    }

                rowRanges[address3(outPos, outSize)] = nonZeroInRow;
            }
        }

    // Convert rowRanges from counts to cumulative counts
    int offset = 0;

	for (int i = 0; i < numOut; i++) {
		int temp = rowRanges[i];

		rowRanges[i] = offset;

		offset += temp;
	}

    rowRanges[numOut] = offset;

    _rows = numOut;
    _columns = inSize.x * inSize.y * inSize.z;
	_numNonZeroValues = nonZeroValues.size();

	_nonZeroValues = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, nonZeroValues.size() * sizeof(cl_float));
	_rowRanges = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, rowRanges.size() * sizeof(cl_int));
	_columnIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, columnIndices.size() * sizeof(cl_int));

	cs.getQueue().enqueueWriteBuffer(_nonZeroValues, CL_TRUE, 0, nonZeroValues.size() * sizeof(cl_float), nonZeroValues.data());
	cs.getQueue().enqueueWriteBuffer(_rowRanges, CL_TRUE, 0, rowRanges.size() * sizeof(cl_int), rowRanges.data());
	cs.getQueue().enqueueWriteBuffer(_columnIndices, CL_TRUE, 0, columnIndices.size() * sizeof(cl_int), columnIndices.data());
}

void SparseMatrix::initT(ComputeSystem &cs) {
	std::vector<cl_int> columnRanges(_columns + 1, 0);
	std::vector<cl_int> rowIndices(_numNonZeroValues);
	std::vector<cl_int> nonZeroValueIndices(_numNonZeroValues);

	std::vector<cl_int> rowRanges(_rows + 1);

	cs.getQueue().enqueueReadBuffer(_rowRanges, CL_TRUE, 0, rowRanges.size() * sizeof(cl_int), rowRanges.data());

	std::vector<cl_int> columnIndices(_numNonZeroValues);

	cs.getQueue().enqueueReadBuffer(_columnIndices, CL_TRUE, 0, columnIndices.size() * sizeof(cl_int), columnIndices.data());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++)
			columnRanges[columnIndices[j]]++;
	}

    // Convert columnRanges from counts to cumulative counts
	int offset = 0;

	for (int i = 0; i < _columns; i++) {
		int temp = columnRanges[i];

		columnRanges[i] = offset;

		offset += temp;
	}

	columnRanges[_columns] = offset;

	std::vector<int> columnOffsets = columnRanges;

	for (int i = 0; i < _rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++) {
			int colIndex = columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			rowIndices[nonZeroIndexT] = i;

			nonZeroValueIndices[nonZeroIndexT] = j;

			columnOffsets[colIndex]++;
		}
	}

	// Copy to buffers
	_columnRanges = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, columnRanges.size() * sizeof(cl_int));
	_rowIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, rowIndices.size() * sizeof(cl_int));
	_nonZeroValueIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, nonZeroValueIndices.size() * sizeof(cl_int));

	cs.getQueue().enqueueWriteBuffer(_columnRanges, CL_TRUE, 0, columnRanges.size() * sizeof(cl_int), columnRanges.data());
	cs.getQueue().enqueueWriteBuffer(_rowIndices, CL_TRUE, 0, rowIndices.size() * sizeof(cl_int), rowIndices.data());
	cs.getQueue().enqueueWriteBuffer(_nonZeroValueIndices, CL_TRUE, 0, nonZeroValueIndices.size() * sizeof(cl_int), nonZeroValueIndices.data());

	_hasT = true;
}

void SparseMatrix::writeToStream(
	ComputeSystem &cs,
	std::ostream &os
) {
	os.write(reinterpret_cast<const char*>(&_rows), sizeof(cl_int));
	os.write(reinterpret_cast<const char*>(&_columns), sizeof(cl_int));
	os.write(reinterpret_cast<const char*>(&_numNonZeroValues), sizeof(cl_int));

	unsigned char hasT = _hasT;

	os.write(reinterpret_cast<const char*>(&hasT), sizeof(unsigned char));

	std::vector<cl_float> nonZeroValues(_numNonZeroValues);
	cs.getQueue().enqueueReadBuffer(_nonZeroValues, CL_TRUE, 0, nonZeroValues.size() * sizeof(cl_float), nonZeroValues.data());
	os.write(reinterpret_cast<const char*>(nonZeroValues.data()), nonZeroValues.size() * sizeof(cl_float));

	std::vector<cl_int> rowRanges(_rows + 1);
	cs.getQueue().enqueueReadBuffer(_rowRanges, CL_TRUE, 0, rowRanges.size() * sizeof(cl_int), rowRanges.data());
	os.write(reinterpret_cast<const char*>(rowRanges.data()), rowRanges.size() * sizeof(cl_int));

	std::vector<cl_int> columnIndices(_numNonZeroValues);
	cs.getQueue().enqueueReadBuffer(_columnIndices, CL_TRUE, 0, columnIndices.size() * sizeof(cl_int), columnIndices.data());
	os.write(reinterpret_cast<const char*>(columnIndices.data()), columnIndices.size() * sizeof(cl_int));

	if (_hasT) {
		std::vector<cl_int> nonZeroValueIndices(_numNonZeroValues);
		cs.getQueue().enqueueReadBuffer(_nonZeroValueIndices, CL_TRUE, 0, nonZeroValueIndices.size() * sizeof(cl_int), nonZeroValueIndices.data());
		os.write(reinterpret_cast<const char*>(nonZeroValueIndices.data()), nonZeroValueIndices.size() * sizeof(cl_int));

		std::vector<cl_int> columnRanges(_columns + 1);
		cs.getQueue().enqueueReadBuffer(_columnRanges, CL_TRUE, 0, columnRanges.size() * sizeof(cl_int), columnRanges.data());
		os.write(reinterpret_cast<const char*>(columnRanges.data()), columnRanges.size() * sizeof(cl_int));

		std::vector<cl_int> rowIndices(_numNonZeroValues);
		cs.getQueue().enqueueReadBuffer(_rowIndices, CL_TRUE, 0, rowIndices.size() * sizeof(cl_int), rowIndices.data());
		os.write(reinterpret_cast<const char*>(rowIndices.data()), rowIndices.size() * sizeof(cl_int));
	}
}

void SparseMatrix::readFromStream(
	ComputeSystem &cs,
	std::istream &is
) {
	is.read(reinterpret_cast<char*>(&_rows), sizeof(cl_int));
	is.read(reinterpret_cast<char*>(&_columns), sizeof(cl_int));
	is.read(reinterpret_cast<char*>(&_numNonZeroValues), sizeof(cl_int));

	unsigned char hasT;
	
	is.read(reinterpret_cast<char*>(&hasT), sizeof(unsigned char));

	_hasT = hasT;

	std::vector<cl_float> nonZeroValues(_numNonZeroValues);
	is.read(reinterpret_cast<char*>(nonZeroValues.data()), nonZeroValues.size() * sizeof(cl_float));
	_nonZeroValues = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, nonZeroValues.size() * sizeof(cl_float));
	cs.getQueue().enqueueWriteBuffer(_nonZeroValues, CL_TRUE, 0, nonZeroValues.size() * sizeof(cl_float), nonZeroValues.data());

	std::vector<cl_int> rowRanges(_rows + 1);
	is.read(reinterpret_cast<char*>(rowRanges.data()), rowRanges.size() * sizeof(cl_int));
	_rowRanges = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, rowRanges.size() * sizeof(cl_int));
	cs.getQueue().enqueueWriteBuffer(_rowRanges, CL_TRUE, 0, rowRanges.size() * sizeof(cl_int), rowRanges.data());

	std::vector<cl_int> columnIndices(_numNonZeroValues);
	is.read(reinterpret_cast<char*>(columnIndices.data()), columnIndices.size() * sizeof(cl_int));
	_columnIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, columnIndices.size() * sizeof(cl_int));
	cs.getQueue().enqueueWriteBuffer(_columnIndices, CL_TRUE, 0, columnIndices.size() * sizeof(cl_int), columnIndices.data());

	if (_hasT) {
		std::vector<cl_int> nonZeroValueIndices(_numNonZeroValues);
		is.read(reinterpret_cast<char*>(nonZeroValueIndices.data()), nonZeroValueIndices.size() * sizeof(cl_int));
		_nonZeroValueIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, nonZeroValueIndices.size() * sizeof(cl_int));
		cs.getQueue().enqueueWriteBuffer(_nonZeroValueIndices, CL_TRUE, 0, nonZeroValueIndices.size() * sizeof(cl_int), nonZeroValueIndices.data());
		
		std::vector<cl_int> columnRanges(_columns + 1);
		is.read(reinterpret_cast<char*>(columnRanges.data()), columnRanges.size() * sizeof(cl_int));
		_columnRanges = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, columnRanges.size() * sizeof(cl_int));
		cs.getQueue().enqueueWriteBuffer(_columnRanges, CL_TRUE, 0, columnRanges.size() * sizeof(cl_int), columnRanges.data());
		
		std::vector<cl_int> rowIndices(_numNonZeroValues);
		is.read(reinterpret_cast<char*>(rowIndices.data()), rowIndices.size() * sizeof(cl_int));
		_rowIndices = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, rowIndices.size() * sizeof(cl_int));
		cs.getQueue().enqueueWriteBuffer(_rowIndices, CL_TRUE, 0, rowIndices.size() * sizeof(cl_int), rowIndices.data());
	}
}