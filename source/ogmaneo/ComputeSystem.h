// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

#include <random>

namespace ogmaneo {
class ComputeSystem {
public:
	ctpl::thread_pool pool;

	// Default batch sizes for dimensions 1-3
	int batchSize1;
	Int2 batchSize2;
	Int3 batchSize3;

	// Default RNG
	std::mt19937 rng;

	ComputeSystem(
		int numWorkers
	)
	: pool(numWorkers), batchSize1(1024), batchSize2(1, 1), batchSize3(1, 1, 1)
	{}
};
} // namespace ogmaneo