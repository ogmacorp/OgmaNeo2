// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"
#include <omp.h>

#include <random>

namespace ogmaneo {
class ComputeSystem {
public:
	// Default batch sizes for dimensions 1-3
	int batchSize1;
	Int2 batchSize2;
	Int3 batchSize3;

	// Default RNG
	std::mt19937 rng;

	ComputeSystem()
	:
	batchSize1(1024),
	batchSize2(2, 2),
	batchSize3(2, 2, 2)
	{}

	static void setNumThreads(int numThreads) {
		omp_set_num_threads(numThreads);
	}
};
} // namespace ogmaneo