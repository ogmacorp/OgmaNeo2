// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld._size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::recon(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2C(pos, Int2(vld._size.x, vld._size.y));

    int targetC = (*inputCs[vli])[visibleColumnIndex];

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3C(Int3(pos.x, pos.y, vc), vld._size);

        float target = (vc == targetC ? 1.0f : 0.0f);

        float sum = vl._weights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._visibleCounts[visibleColumnIndex]);

        vl._reconErrors[visibleIndex] = _alpha * (target - sum);
    }
}

void SparseCoder::learn(
    const Int2 &pos,
    std::mt19937 &rng
) {
    int hiddenColumnIndex = address2C(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int hiddenIndexMax = address3C(Int3(pos.x, pos.y, _hiddenCs[hiddenColumnIndex]), _hiddenSize);

    if (_refractoryTimers[hiddenIndexMax] == 0) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            vl._weights.hebbErrors(vl._reconErrors, hiddenIndexMax);
        }
    }

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3C(Int3(pos.x, pos.y, hc), _hiddenSize);

        if (_refractoryTimers[hiddenIndex] > 0)
            _refractoryTimers[hiddenIndex]--;
    }

    _refractoryTimers[hiddenIndexMax] = _refractoryTicks;
}

void SparseCoder::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(0.999f, 1.0f);

    _hiddenCounts = IntBuffer(numHiddenColumns, 0);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld._size, _hiddenSize, vld._radius, vl._weights);

        for (int i = 0; i < vl._weights._nonZeroValues.size(); i++)
            vl._weights._nonZeroValues[i] = weightDist(cs._rng);

        // Generate transpose (needed for reconstruction)
        vl._weights.initT();

        vl._reconErrors = FloatBuffer(numVisible, 0.0f);

        // Counts
        for (int i = 0; i < numHiddenColumns; i++)
            _hiddenCounts[i] += vl._weights.counts(i * _hiddenSize.z) / vld._size.z;
            
        vl._visibleCounts = IntBuffer(numVisibleColumns);

        for (int i = 0; i < numVisibleColumns; i++)
            vl._visibleCounts[i] = vl._weights.countsT(i * vld._size.z) / _hiddenSize.z;
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);

    _refractoryTimers = IntBuffer(numHidden, 0);
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputCs);
#else
    runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    recon(Int2(x, y), cs._rng, inputCs, vli);
#else
            runKernel2(cs, std::bind(SparseCoder::reconKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                learn(Int2(x, y), cs._rng);
#else
        runKernel2(cs, std::bind(SparseCoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
    }
}

void SparseCoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_refractoryTicks), sizeof(int));

    writeBufferToStream(os, &_hiddenCs);
    writeBufferToStream(os, &_hiddenCounts);
    writeBufferToStream(os, &_refractoryTimers);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._reconErrors);

        writeBufferToStream(os, &vl._visibleCounts);
    }
}

void SparseCoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_refractoryTicks), sizeof(int));

    readBufferFromStream(is, &_hiddenCs);
    readBufferFromStream(is, &_hiddenCounts);
    readBufferFromStream(is, &_refractoryTimers);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._reconErrors);

        readBufferFromStream(is, &vl._visibleCounts);
    }
}