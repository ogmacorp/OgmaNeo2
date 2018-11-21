// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ThreadPool.h"

#include <random>

namespace ogmaneo {
	/*!
	\brief Compute system. Mainly passed to other functions. Contains thread pooling and random number generator information
	*/
    class ComputeSystem {
	public:
		/*!
		\brief System thread pool
		*/
		ThreadPool _pool;

		//!@{
		/*!
		\brief Default batch sizes
		*/
		int _batchSize1;
		Int2 _batchSize2;
		Int3 _batchSize3;
		//!@}

		/*!
		\brief Initialize the system
		\param numWorkers number of thread pool worker threads
		*/
        ComputeSystem(size_t numWorkers)
		: _batchSize1(256), _batchSize2(16, 16), _batchSize3(4, 4, 4)
		{
			_pool.create(numWorkers);
		}
    };
}