// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
	ctpl::thread_pool _pool;

	// Default batch sizes for dimensions 1-3
	int _batchSize1;
	Int2 _batchSize2;
	Int3 _batchSize3;

	// Default RNG
	std::mt19937 _rng;

	ComputeSystem(
		int numWorkers
	)
	: _pool(numWorkers), _batchSize1(1024), _batchSize2(1, 1), _batchSize3(1, 1, 1)
	{}
};
} // namespace ogmaneo