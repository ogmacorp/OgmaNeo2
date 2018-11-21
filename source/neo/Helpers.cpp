// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

using namespace ogmaneo;

void KernelWorkItem1::run() {
    for (int x = 0; x < _batchSize; x++) {
        int bPos = _pos + x;

        _func(bPos, _rng);
    }
}

void KernelWorkItem2::run() {
    for (int x = 0; x < _batchSize.x; x++)
        for (int y = 0; y < _batchSize.y; y++) {
            Int2 bPos;
            bPos.x = _pos.x + x;
            bPos.y = _pos.y + y;

            _func(bPos, _rng);
        }
}

void KernelWorkItem3::run() {
    for (int x = 0; x < _batchSize.x; x++)
        for (int y = 0; y < _batchSize.y; y++)
            for (int z = 0; z < _batchSize.z; z++) {
                Int3 bPos;
                bPos.x = _pos.x + x;
                bPos.y = _pos.y + y;
                bPos.z = _pos.z + z;

                _func(bPos, _rng);
            }
}

void ogmaneo::runKernel1(ComputeSystem &cs, std::function<void(int, std::mt19937 &rng)> &func, int size, std::mt19937 &rng, int batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    int batches = (size + batchSize - 1) / batchSize;
    
    for (int x = 0; x < batches; x++) {
        std::shared_ptr<KernelWorkItem1> kwi = std::make_shared<KernelWorkItem1>();

        kwi->_func = func;
        kwi->_pos = x * batchSize;
        kwi->_batchSize = std::min(size - x * batchSize, batchSize);
        kwi->_rng.seed(seedDist(rng));

        cs._pool.addItem(kwi);
    }

    cs._pool.wait();
}

void ogmaneo::runKernel2(ComputeSystem &cs, std::function<void(const Int2 &, std::mt19937 &rng)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int2 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y);

    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) {
            std::shared_ptr<KernelWorkItem2> kwi = std::make_shared<KernelWorkItem2>();

            kwi->_func = func;
            kwi->_pos.x = x;
            kwi->_pos.y = y;
            kwi->_batchSize = Int2(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y));
            kwi->_rng.seed(seedDist(rng));

            cs._pool.addItem(kwi);
        }

    cs._pool.wait();
}

void ogmaneo::runKernel3(ComputeSystem &cs, std::function<void(const Int3 &, std::mt19937 &rng)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    Int3 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y, (size.z + batchSize.z - 1) / batchSize.z);

    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) 
            for (int z = 0; z < batches.z; z++) {
                std::shared_ptr<KernelWorkItem3> kwi = std::make_shared<KernelWorkItem3>();

                kwi->_func = func;
                kwi->_pos.x = x;
                kwi->_pos.y = y;
                kwi->_batchSize = Int3(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y), std::min(size.z - z * batchSize.z, batchSize.z));
                kwi->_rng.seed(seedDist(rng));

                cs._pool.addItem(kwi);
            }

    cs._pool.wait();
}