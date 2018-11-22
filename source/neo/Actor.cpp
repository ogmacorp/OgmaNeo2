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

void Actor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs) {
    // Value
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);
    
    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * (_hiddenSize.z + 1);

    int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

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

        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputs[vli])[address2(visiblePosition, vld._size.x)];

                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                value += vl._weights[dPartial + az * dxyz];
            }

        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    value /= std::max(1.0f, count);

    int hiddenIndex = address2(pos, _hiddenSize.x);

    _hiddenValues[hiddenIndex] = value;

    // Action
    std::vector<float> hiddenActivations(_hiddenSize.z);
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        float sum = 0.0f;

        Int3 actionHiddenPosition(pos.x, pos.y, hc);

        int dActionPartial = actionHiddenPosition.x + actionHiddenPosition.y * _hiddenSize.x + actionHiddenPosition.z * dxy;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

            Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

            int diam = vld._radius * 2 + 1;
            int diam2 = diam * diam;

            Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
            Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

            for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
                for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                    Int2 visiblePosition(x, y);

                    int visibleC = (*inputs[vli])[address2(visiblePosition, vld._size.x)];

                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                    sum += vl._weights[dActionPartial + az * dxyz];
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

    std::uniform_real_distribution<float> cuspDist(0.0f, total);

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

    _hiddenCs[hiddenIndex] = selectIndex;
}

void Actor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputsPrev, const IntBuffer* hiddenCsPrev, float q, float g) {
    // New Q
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);

    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * (_hiddenSize.z + 1);

    int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

    float valuePrev = 0.0f;
    float countPrev = 0.0f;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputsPrev[vli])[address2(visiblePosition, vld._size.x)];

                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                valuePrev += vl._weights[dPartial + az * dxyz];
            }

        countPrev += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    valuePrev /= std::max(1.0f, countPrev);

    int hiddenIndex = address2(pos, _hiddenSize.x);

    float tdError = q + g * _hiddenValues[hiddenIndex] - valuePrev;
   
    float alphaTdError = _alpha * tdError;
    float betaTdError = _beta * tdError;

    int actionIndex = (*hiddenCsPrev)[hiddenIndex];

    int dActionPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + actionIndex * dxy;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputsPrev[vli])[address2(visiblePosition, vld._size.x)];

                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                vl._weights[dPartial + az * dxyz] += alphaTdError;
                vl._weights[dActionPartial + az * dxyz] += betaTdError;
            }

    }
}

void Actor::createRandom(ComputeSystem &cs,
    Int3 hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
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

        runKernel1(cs, std::bind(Actor::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSize, cs._rng, cs._batchSize1);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);

    _hiddenValues = FloatBuffer(numHiddenColumns);

    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenValues, 0.0f), numHiddenColumns, cs._rng, cs._batchSize1);

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

void Actor::step(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs, float reward, bool learn) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);

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
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, visibleCs[vli], s._visibleCs[vli].get()), numVisibleColumns, cs._rng, cs._batchSize1);
        }

        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, s._hiddenCs.get()), numHiddenColumns, cs._rng, cs._batchSize1);

        s._reward = reward;
    }

    // Learn
    if (learn && _historySize > 1) {
        const HistorySample &sPrev = _historySamples[0];

        float q = 0.0f;

        for (int t = _historySize - 1; t >= 1; t--)
            q += _historySamples[t]._reward * std::pow(_gamma, t - 1);

        float g = std::pow(_gamma, _historySize - 1);

        runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, constGet(sPrev._visibleCs), sPrev._hiddenCs.get(), q, g), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
    }
}