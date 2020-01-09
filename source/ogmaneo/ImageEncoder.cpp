// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

#include <algorithm>

using namespace ogmaneo;

bool pairfiCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first; // Backwards so largest is in front
}

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    std::vector<std::pair<float, int>> activations(_hiddenSize.z);

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum -= vl._weights.distance2(*inputActs[vli], hiddenIndex);
            count += vl._weights.count(hiddenIndex);
        }

        sum /= std::max(1, count);

        activations[hc] = std::make_pair(sum, hc);

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        std::sort(activations.begin(), activations.end(), pairfiCompare);

        for (int i = 0; i < _hiddenSize.z; i++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, activations[i].second), _hiddenSize);

            float strength = std::exp(-i * i * _gamma / std::max(0.001f, _hiddenResources[hiddenIndex])) * _hiddenResources[hiddenIndex];

            _hiddenResources[hiddenIndex] -= _alpha * strength;

            // For each visible layer
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                vl._weights.hebb(*inputActs[vli], hiddenIndex, strength);
            }
        }
    }
}

void ImageEncoder::backward(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCs,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float sum = vl._weights.multiplyOHVsT(*hiddenCs, visibleIndex, _hiddenSize.z) / std::max(1, vl._weights.countT(visibleIndex) / _hiddenSize.z);

        vl._reconActs[visibleIndex] = sum;
    }
}

void ImageEncoder::initRandom(
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

    std::normal_distribution<float> weightDist(0.0f, 1.0f);

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

        vl._reconActs = FloatBuffer(numVisible, 0.0f);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);

    _hiddenResources = FloatBuffer(numHidden, 1.0f);
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActs, learnEnabled);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActs, learnEnabled), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void ImageEncoder::reconstruct(
    ComputeSystem &cs,
    const IntBuffer* hiddenCs
) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                backward(Int2(x, y), cs._rng, hiddenCs, vli);
#else
        runKernel2(cs, std::bind(ImageEncoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
    }
}

void ImageEncoder::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(float));

    writeBufferToStream(os, &_hiddenCs);
    writeBufferToStream(os, &_hiddenResources);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);
    readBufferFromStream(is, &_hiddenResources);

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

        vl._reconActs = FloatBuffer(numVisible, 0.0f);
    }
}