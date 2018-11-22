// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

#include "ComputeSystem.h"

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

void ogmaneo::runKernel1(ComputeSystem &cs, const std::function<void(int, std::mt19937 &rng)> &func, int size, std::mt19937 &rng, int batchSize) {
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

void ogmaneo::runKernel2(ComputeSystem &cs, const std::function<void(const Int2 &, std::mt19937 &rng)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int2 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y);

    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) {
            std::shared_ptr<KernelWorkItem2> kwi = std::make_shared<KernelWorkItem2>();

            kwi->_func = func;
            kwi->_pos.x = x * batchSize.x;
            kwi->_pos.y = y * batchSize.y;
            kwi->_batchSize = Int2(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y));
            kwi->_rng.seed(seedDist(rng));

            cs._pool.addItem(kwi);
        }

    cs._pool.wait();
}

void ogmaneo::runKernel3(ComputeSystem &cs, const std::function<void(const Int3 &, std::mt19937 &rng)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    Int3 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y, (size.z + batchSize.z - 1) / batchSize.z);

    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) 
            for (int z = 0; z < batches.z; z++) {
                std::shared_ptr<KernelWorkItem3> kwi = std::make_shared<KernelWorkItem3>();

                kwi->_func = func;
                kwi->_pos.x = x * batchSize.x;
                kwi->_pos.y = y * batchSize.y;
                kwi->_pos.z = z * batchSize.z;
                kwi->_batchSize = Int3(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y), std::min(size.z - z * batchSize.z, batchSize.z));
                kwi->_rng.seed(seedDist(rng));

                cs._pool.addItem(kwi);
            }

    cs._pool.wait();
}

void ogmaneo::fillInt(int pos, std::mt19937 &rng, IntBuffer* buffer, int fillValue) {
    (*buffer)[pos] = fillValue;
}

void ogmaneo::fillFloat(int pos, std::mt19937 &rng, FloatBuffer* buffer, float fillValue) {
    (*buffer)[pos] = fillValue;
}

void ogmaneo::copyInt(int pos, std::mt19937 &rng, const IntBuffer* src, IntBuffer* dst) {
    (*dst)[pos] = (*src)[pos];
}

void ogmaneo::copyFloat(int pos, std::mt19937 &rng, const FloatBuffer* src, FloatBuffer* dst) {
    (*dst)[pos] = (*src)[pos];
}

bool ogmaneo::inBounds0(const Int2 &position, const Int2 &upperBound) {
    return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool ogmaneo::inBounds(const Int2 &position, const Int2 &lowerBound, const Int2 &upperBound) {
    return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

Int2 ogmaneo::project(const Int2 &position, const Float2 &toScalars) {
    return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
}

Int2 ogmaneo::projectf(const Float2 &position, const Float2 &toScalars) {
    return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
}

int ogmaneo::address2(const Int2 &pos, int dim) {
    return pos.x + pos.y * dim;
}

int ogmaneo::address3(const Int3 &pos, const Int2 &dims) {
    return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
}

int ogmaneo::address4(const Int4 &pos, const Int3 &dims) {
    int dxy = dims.x * dims.y;
    int dxyz = dxy * dims.z;

    return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
}

std::vector<IntBuffer*> ogmaneo::get(const std::vector<std::shared_ptr<IntBuffer>> &v) {
    std::vector<IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<FloatBuffer*> ogmaneo::get(const std::vector<std::shared_ptr<FloatBuffer>> &v) {
    std::vector<FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<const IntBuffer*> ogmaneo::constGet(const std::vector<std::shared_ptr<IntBuffer>> &v) {
    std::vector<const IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<const FloatBuffer*> ogmaneo::constGet(const std::vector<std::shared_ptr<FloatBuffer>> &v) {
    std::vector<const FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}