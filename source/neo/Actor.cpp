// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

// Kernels
void Actor::init(int pos, std::mt19937 &rng, int vli) {
	std::uniform_real_distribution<float> weightDist(-0.0001f, 0.0001f);

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void Actor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<IntBuffer*> &inputs) {
    // Value
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);

    float value = 0.0f;
    float count = 0.0f;

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

                    value += vl._weights[address4(wPos, _hiddenSize)];
                    count += 1.0f;
                }
            }
    }

    _hiddenValues[hiddenIndex] = value / std::max(1.0f, count);

    // Action
    std::vector<float> hiddenActivations(_hiddenSize.z);
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        float sum = 0.0f;

        Int3 actionHiddenPosition(pos.x, pos.y, hc);

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

                        Int4 wPos(actionHiddenPosition.x, actionHiddenPosition.y, actionHiddenPosition.z, offset.x + offset.y * diam + visibleC * diam2);
    
                        sum += vl._weights[address4(wPos, _hiddenSize)];
                    }
                }
        }

        hiddenActivations[hc] = sum / std::max(1.0f, count);

        maxActivation = std::max(maxActivation, hiddenActivations[hc]);
    }

    // Boltzmann exploration
    float total = 0.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        hiddenActivations[hc] = std::exp(hiddenActivations[hc] - maxActivation);
        total += hiddenActivations[hc];
    }

    std::uniform_real_distribution<float> cuspDist(0, total);

    float cusp = cuspDist(rng);

    float sumSoFar = 0.0f;
    int selectIndex = 0;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        sumSoFar += hiddenActivations[hc];

        if (sumSoFar >= cusp) {
            selectIndex = hc;
            break;
        }
    }

    int hiddenIndex = address2(pos, _hiddenSize.x);

    _hiddenCs[hiddenIndex] = selectIndex;
}

void Actor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<std::shared_ptr<IntBuffer>> &inputsPrev, IntBuffer* hiddenCsPrev, float q, float g) {
    // New Q
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);

    float valuePrev = 0.0f;

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

                    int visibleC = (*inputsPrev[vli])[visibleIndex];

                    Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                    Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z, offset.x + offset.y * diam + visibleC * diam2);

                    valuePrev += vl._weights[address4(wPos, _hiddenSize)];
                }
            }
    }

    int hiddenIndex = address2(pos, _hiddenSize.x);

    float tdError = q + g * _hiddenValues[hiddenIndex] - valuePrev;

    float alphaTdError = _alpha * tdError;
    float betaTdError = _beta * tdError;

    int actionIndex = (*hiddenCsPrev)[hiddenIndex];

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

                    int visibleC = (*inputsPrev[vli])[visibleIndex];

                    Int2 offset(visiblePosition.x - fieldLowerBound.x, visiblePosition.y - fieldLowerBound.y);

                    Int4 valueWPos(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z, offset.x + offset.y * diam + visibleC * diam2);
                    Int4 actionWPos(hiddenPosition.x, hiddenPosition.y, actionIndex, valueWPos.w);

                    vl._weights[address4(valueWPos, _hiddenSize)] += alphaTdError;
                    vl._weights[address4(actionWPos, _hiddenSize)] += betaTdError;
                }
            }
    }
}

void Actor::createRandom(ComputeSystem &cs,
    Int3 hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;
    int numHidden1 = numHiddenColumns * (_hiddenSize.z + 1);

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSize = numHidden1 * numWeightsPerHidden;

        // Set context for kernel call
        vl._weights = FloatBuffer(weightsSize);

        runKernel1(cs, std::bind(init, std::placeholders::_1, std::placeholders::_2, vli), weightsSize, rng, cs._batchSize1);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, rng, cs._batchSize1);

    _hiddenValues = FloatBuffer(numHiddenColumns);

    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenValues, 0.0f), numHiddenColumns, rng, cs._batchSize1);

    // History samples
    _historySize = 0;
    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i]._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            _historySamples[i]._visibleCs[vli] = std::make_shared<IntBuffer>(numVisibleColumns);
        }

        _historySamples[i]._hiddenCs = std::make_shared<IntBuffer>(numHiddenColumns);
    }
}

void Actor::step(ComputeSystem &cs, const std::vector<IntBuffer*> &visibleCs, std::mt19937 &rng, float reward, bool learn) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;
    int numHidden1 = numHiddenColumns * (_hiddenSize.z + 1);

    runKernel2(cs, std::bind(foward, std::placeholders::_1, std::placeholders::_2, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), rng, cs._batchSize2);

    // Add sample
    if (_historySize == _historySamples.size()) {
        // Circular buffer swap
        HistorySample temp = _historySamples.front();

        for (int i = 0; i < _historySamples.size() - 1; i++) {
            _historySamples[i] = _historySamples[i + 1];
        }

        _historySamples.back() = temp;
    }

    if (_historySize < _historySamples.size())
        _historySize++;
    
    {
        HistorySample &s = _historySamples[_historySize - 1];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            // Copy visible Cs
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &visibleCs[vli], s._visibleCs[vli].get()), numVisibleColumns, rng, cs._batchSize1);
        }

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, s._hiddenCs.get()), numHiddenColumns, rng, cs._batchSize1);

        s._reward = reward;
    }

    // Learn
    if (learn && _historySize > 1) {
        const HistorySample &sPrev = _historySamples[0];

        cl_float q = 0.0f;

        for (int t = _historySize - 1; t >= 1; t--)
            q += _historySamples[t]._reward * std::pow(_gamma, t - 1);

        cl_float g = std::pow(_gamma, _historySize - 1);

        runKernel2(cs, std::bind(learn, std::placeholders::_1, std::placeholders::_2, sPrev._visibleCs, sPrev._hiddenCs.get(), q, g), Int2(_hiddenSize.x, _hiddenSize.y), rng, cs._batchSize2);
    }
}