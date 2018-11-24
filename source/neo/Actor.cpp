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
    // Randomly initialize weights in range
	std::uniform_real_distribution<float> weightDist(-0.0001f, 0.0001f);

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void Actor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs) {
    // Value
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);
    
    // Cache address calculations (taken from addressN functions)
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * (_hiddenSize.z + 1);

    // Partially computed address of weight
    int dValuePartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

    // ------------------------------ Value ------------------------------

    // Accumulator values
    float value = 0.0f;
    float count = 0.0f;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputs[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                value += vl._weights[dValuePartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
            }

        // Count can be computed outside of loop, this is the value equavilent to count += 1.0f after each value increment
        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    // Normalize
    value /= std::max(1.0f, count);

    int hiddenIndex = address2(pos, _hiddenSize.x);

    // Set the value for later
    _hiddenValues[hiddenIndex] = value;

    // ------------------------------ Action ------------------------------

    std::vector<float> hiddenActivations(_hiddenSize.z);
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        float sum = 0.0f;

        Int3 actionHiddenPosition(pos.x, pos.y, hc);

        // Partially computed address of weight
        int dActionPartial = actionHiddenPosition.x + actionHiddenPosition.y * _hiddenSize.x + actionHiddenPosition.z * dxy;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            // Center of projected position
            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

            // Lower corner
            Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

            // Additional addressing dimensions
            int diam = vld._radius * 2 + 1;
            int diam2 = diam * diam;

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
            Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

            for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
                for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                    Int2 visiblePosition(x, y);

                    int visibleC = (*inputs[vli])[address2(visiblePosition, vld._size.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                    sum += vl._weights[dActionPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }
        }

        // Normalize and save value for later
        hiddenActivations[hc] = sum / std::max(1.0f, count);

        // Determine highest value
        maxActivation = std::max(maxActivation, hiddenActivations[hc]);
    }

    // Boltzmann exploration with roulette wheel
    float total = 0.0f;

    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        hiddenActivations[hc] = std::exp(hiddenActivations[hc] - maxActivation); // Subtract max for numerical stability
        total += hiddenActivations[hc];
    }

    // Random cusp for roulette wheel selection
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

    // Selected action output
    _hiddenCs[hiddenIndex] = selectIndex;
}

void Actor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputsPrev, const IntBuffer* hiddenCsPrev, float q, float g) {
    // New Q
    Int3 hiddenPosition(pos.x, pos.y, _hiddenSize.z);

    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * (_hiddenSize.z + 1);

    // Partially computed address of weight
    int dValuePartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

    // As in forward, compute value and count form normalization, but based on previous (historyCapacity timesteps ago typically) visible and hidden states
    float valuePrev = 0.0f;
    float countPrev = 0.0f;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputsPrev[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                valuePrev += vl._weights[dValuePartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
            }

        // Count can be computed outside of loop, this is the value equavilent to count += 1.0f after each value increment
        countPrev += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    // Normalize
    valuePrev /= std::max(1.0f, countPrev);

    int hiddenIndex = address2(pos, _hiddenSize.x);

    // Temporal difference error
    float tdError = q + g * _hiddenValues[hiddenIndex] - valuePrev;
   
    // Deltas for value and action
    float alphaTdError = _alpha * tdError;
    float betaTdError = _beta * tdError;

    // Selected (past) action index
    int actionIndex = (*hiddenCsPrev)[hiddenIndex];

    // Partially computed address, this time for action
    int dActionPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + actionIndex * dxy;

    // For each visible layer
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Center of projected position
        Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

        // Lower corner
        Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

        // Additional addressing dimensions
        int diam = vld._radius * 2 + 1;
        int diam2 = diam * diam;

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
        Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 visiblePosition(x, y);

                int visibleC = (*inputsPrev[vli])[address2(visiblePosition, vld._size.x)];

                // Final component of address
                int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                // Update both value and action using cached parts to compute weight addressed (equivalent to calling address4)
                vl._weights[dValuePartial + az * dxyz] += alphaTdError;
                vl._weights[dActionPartial + az * dxyz] += betaTdError;
            }
    }
}

void Actor::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;
    int numHidden1 = numHiddenColumns * (_hiddenSize.z + 1); // Expanded dimension used for weight matrix, so includes an additional slice for the value function (rest is all action)

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        // Projection constant
        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSize = numHidden1 * numWeightsPerHidden;

        // Create weight matrix for this visible layer and initialize randomly
        vl._weights = FloatBuffer(weightsSize);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < weightsSize; x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(Actor::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSize, cs._rng, cs._batchSize1);
#endif
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Hidden values
    _hiddenValues = FloatBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillFloat(x, cs._rng, &_hiddenValues, 0.0f);
#else
    runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &_hiddenValues, 0.0f), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Create (pre-allocated) history samples
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

void Actor::step(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs, float reward, bool learnEnabled) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs);
#else
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

    // Add sample
    if (_historySize == _historySamples.size()) {
        // Circular buffer swap
        HistorySample temp = _historySamples.front();

        for (int i = 0; i < _historySamples.size() - 1; i++) {
            _historySamples[i] = _historySamples[i + 1];
        }

        _historySamples.back() = temp;
    }

    // If not at cap, increment
    if (_historySize < _historySamples.size())
        _historySize++;
    
    // Add new sample
    {
        HistorySample &s = _historySamples[_historySize - 1];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            // Copy visible Cs
#ifdef KERNEL_DEBUG
            for (int x = 0; x < numVisibleColumns; x++)
                copyInt(x, cs._rng, visibleCs[vli], s._visibleCs[vli].get());
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, visibleCs[vli], s._visibleCs[vli].get()), numVisibleColumns, cs._rng, cs._batchSize1);
#endif
        }

        // Copy hidden Cs
#ifdef KERNEL_DEBUG
        for (int x = 0; x < numHiddenColumns; x++)
            copyInt(x, cs._rng, &_hiddenCs, s._hiddenCs.get());
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, s._hiddenCs.get()), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

        s._reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && _historySize > 1) {
        const HistorySample &sPrev = _historySamples[0];

        // Compute (partial) Q value, rest is completed in the kernel
        float q = 0.0f;

        for (int t = _historySize - 1; t >= 1; t--)
            q += _historySamples[t]._reward * std::pow(_gamma, t - 1);

        // Discount factor for remainder of Q value
        float g = std::pow(_gamma, _historySize - 1);

        // Learn kernel
#ifdef KERNEL_DEBUG
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                learn(Int2(x, y), cs._rng, constGet(sPrev._visibleCs), sPrev._hiddenCs.get(), q, g);
#else
        runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, constGet(sPrev._visibleCs), sPrev._hiddenCs.get(), q, g), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
    }
}