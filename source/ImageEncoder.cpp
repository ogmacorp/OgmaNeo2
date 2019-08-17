// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

void ImageEncoder::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations
) {
    int hiddenColumnIndex = address2(pos, Int2(_hiddenSize.x, _hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), _hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            sum += vl._weights.multiply(*inputActivations[vli], hiddenIndex);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    _hiddenCs[hiddenColumnIndex] = maxIndex;
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

        float recon = vl._weights.multiplyOHVsT(*hiddenCs, visibleIndex, _hiddenSize.z) / static_cast<float>(vl._visibleCounts[visibleColumnIndex]);
        
        vl._visibleActivations[visibleIndex] = (recon > 0.0f ? 1.0f + recon : std::exp(recon));
    }
}

void ImageEncoder::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const FloatBuffer*> &inputActivations,
    int vli
) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, Int2(vld._size.x, vld._size.y));

    for (int vc = 0; vc < vld._size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld._size);

        float recon = vl._weights.multiplyOHVsT(_hiddenCs, visibleIndex, _hiddenSize.z) / static_cast<float>(vl._visibleCounts[visibleColumnIndex]);

        vl._weights.deltaOHVsT(_hiddenCs, _alpha * ((*inputActivations[vli])[visibleIndex] - (recon > 0.0f ? 1.0f + recon : std::exp(recon))), visibleIndex, _hiddenSize.z);
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

    std::uniform_real_distribution<float> weightDist(0.0f, 0.01f);

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

        vl._visibleActivations = FloatBuffer(numVisible, 0.0f);

        vl._visibleCounts = IntBuffer(numVisibleColumns);

        for (int i = 0; i < numVisibleColumns; i++)
            vl._visibleCounts[i] = std::max(1, vl._weights.countsT(i * vld._size.z) / _hiddenSize.z);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void ImageEncoder::step(
    ComputeSystem &cs,
    const std::vector<const FloatBuffer*> &inputActivations,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;

#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActivations);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_NOTHREAD
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    learn(Int2(x, y), cs._rng, inputActivations, vli);
#else
            runKernel2(cs, std::bind(ImageEncoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
        }
    }
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

    writeBufferToStream(os, &_hiddenCs);

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        const VisibleLayer &vl = _visibleLayers[vli];
        const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl._weights);

        writeBufferToStream(os, &vl._visibleActivations);

        writeBufferToStream(os, &vl._visibleCounts);
    }
}

void ImageEncoder::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));

    readBufferFromStream(is, &_hiddenCs);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        readSMFromStream(is, vl._weights);

        readBufferFromStream(is, &vl._visibleActivations);

        readBufferFromStream(is, &vl._visibleCounts);
    }
}