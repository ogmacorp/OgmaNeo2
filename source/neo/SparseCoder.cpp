// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

// Kernels
void SparseCoder::init(int pos, std::mt19937 &rng, int vli) {
	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void SparseCoder::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<IntBuffer*> &inputs, bool firstStep) {
    int maxIndex = 0;
    float maxValue = -999999.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

            Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

            int diam = vld._radius * 2 + 1;
            int diam2 = diam * diam;

            for (int dx = -vld._radius; dx <= vld._radius; dx++)
                for (int dy = -vld._radius; dy <= vld._radius; dy++) {
                    Int2 visiblePosition(visiblePositionCenter.x + dx, visiblePositionCenter.y + dy);

                    if (inBounds0(visiblePosition, Int2(vld._size.x, vld._size.y))) {
                        int visibleIndex = address2(visiblePosition, vld._size.x);

                        int visibleC = (*inputs[vli])[visibleIndex];

                        Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                        Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z, offset.x + offset.y * diam + visibleC * diam2);

                        sum += std::max(0.0f, vl._weights[address4(wPos, _hiddenSize)] - vl._visibleActivations[visibleIndex]);
                    }
                }
        }

        int hiddenIndex = address3(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y);

        if (firstStep)
            _hiddenActivations[hiddenIndex] = sum;
        else
            _hiddenActivations[hiddenIndex] += sum;

        if (hiddenActivations[hiddenIndex] > maxValue) {
            maxValue = hiddenActivations[hiddenIndex];

            maxIndex = hc;
        }
    }

    _hiddenCs[address2(pos, _hiddenSize.x)] = maxIndex;
}

void SparseCoder::backward(const Int2 &pos, std::mt19937 &rng, const std::vector<IntBuffer*> &inputs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, vld._size.x);

    Int3 visiblePosition(pos.x, pos.y, (*inputs)[visibleColumnIndex]);

    float sum = 0.0f;
    float count = 0.0f;

    for (int dx = -vld._reverseRadii.x; dx <= vld._reverseRadii.x; dx++)
        for (int dy = -vld._reverseRadii.y; dy <= vld._reverseRadii.y; dy++) {
            Int2 hiddenPosition(hiddenPositionCenter.x + dx, hiddenPositionCenter.y + dy);

            if (inBounds0(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y))) {
                // Next layer node's receptive field
                Int2 visibleFieldCenter = project(hiddenPosition, vld._hiddenToVisible);

                Int2 fieldLowerBound(visibleFieldCenter.x - radius, visibleFieldCenter.y - radius);
                Int2 fieldUpperBound(visibleFieldCenter.y + radius + 1, visibleFieldCenter.y + radius + 1);

                // Check for containment
                if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                    int hiddenC = _hiddenCs[address2(hiddenPosition, hiddenSize.x)];

                    Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                    Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, offset.x + offset.y * diam + visiblePosition.z * diam2);

                    sum += vl._weights[address4(wPos, _hiddenSize)];
                    count += 1.0f;
                }
            }
        }

    vl._visibleActivations[visibleColumnIndex] = sum / std::max(1.0f, count);
}

void SparseCoder::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<IntBuffer*> &inputs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleColumnIndex = address2(pos, vld._size.x);

    int inputC = (*inputs)[visibleColumnIndex];

    for (int vc = 0; vc < vld._size.z; vc++) {
        Int3 visiblePosition(pos.x, pos.y, vc);

        float target = (vc == inputC ? 1.0f : 0.0f);

        float sum = 0.0f;
        float count = 0.0f;

        for (int dx = -vld._reverseRadii.x; dx <= vld._reverseRadii.x; dx++)
            for (int dy = -vld._reverseRadii.y; dy <= vld._reverseRadii.y; dy++) {
                Int2 hiddenPosition(hiddenPositionCenter.x + dx, hiddenPositionCenter.y + dy);

                if (inBounds0(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y))) {
                    // Next layer node's receptive field
                    Int2 visibleFieldCenter = project(hiddenPosition, vld._hiddenToVisible);

                    Int2 fieldLowerBound(visibleFieldCenter.x - radius, visibleFieldCenter.y - radius);
                    Int2 fieldUpperBound(visibleFieldCenter.y + radius + 1, visibleFieldCenter.y + radius + 1);

                    // Check for containment
                    if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                        int hiddenC = _hiddenCs[address2(hiddenPosition, hiddenSize.x)];

                        Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                        Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, offset.x + offset.y * diam + visiblePosition.z * diam2);

                        sum += vl._weights[address4(wPos, _hiddenSize)];
                        count += 1.0f;
                    }
                }
            }

        float activation = sum / std::max(1.0f, count);

        float delta = _alpha * (target - activation);

        for (int dx = -vld._reverseRadii.x; dx <= vld._reverseRadii.x; dx++)
            for (int dy = -vld._reverseRadii.y; dy <= vld._reverseRadii.y; dy++) {
                Int2 hiddenPosition(hiddenPositionCenter.x + dx, hiddenPositionCenter.y + dy);

                if (inBounds0(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y))) {
                    // Next layer node's receptive field
                    Int2 visibleFieldCenter = project(hiddenPosition, vld._hiddenToVisible);

                    Int2 fieldLowerBound(visibleFieldCenter.x - radius, visibleFieldCenter.y - radius);
                    Int2 fieldUpperBound(visibleFieldCenter.y + radius + 1, visibleFieldCenter.y + radius + 1);

                    // Check for containment
                    if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                        int hiddenC = _hiddenCs[address2(hiddenPosition, hiddenSize.x)];

                        Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                        Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, offset.x + offset.y * diam + visiblePosition.z * diam2);

                        vl._weights[address4(wPos, _hiddenSize)] += delta;
                    }
                }
            }
    }
}

void SparseCoder::createRandom(ComputeSystem &cs,
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._visibleToHidden = Float2(static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y));

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        vl._reverseRadii = Int2(static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSize = numHidden * numWeightsPerHidden;

        vl._weights = FloatBuffer(weightsSize);

        runKernel1(cs, std::bind(init, std::placeholders::_1, std::placeholders::_2, vli), weightsSize, rng, cs._batchSize1);

        vl._visibleActivations = FloatBuffer(numVisible);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, rng, cs._batchSize1);

    // Hidden activations
    _hiddenActivations = FloatBuffer(numHidden);
}

void SparseCoder::activate(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    for (int it = 0; it < _explainIters; it++) {
        runKernel2(cs, std::bind(foward, std::placeholders::_1, std::placeholders::_2, visibleCs, it == 0), Int2(_hiddenSize.x, _hiddenSize.y), rng, cs._batchSize2);

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            runKernel2(cs, std::bind(backward, std::placeholders::_1, std::placeholders::_2, visibleCs, vli), Int2(vld._size.x, vld._size.y), rng, cs._batchSize2);
        }
    }
}

void SparseCoder::learn(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        runKernel2(cs, std::bind(learn, std::placeholders::_1, std::placeholders::_2, visibleCs, vli), Int2(vld._size.x, vld._size.y), rng, cs._batchSize2);
    }
}