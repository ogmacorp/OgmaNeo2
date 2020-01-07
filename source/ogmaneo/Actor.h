// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace ogmaneo {
// A reinforcement learning layer
class Actor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Visible/input size

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        _size(4, 4, 16),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _valueWeights; // Value function weights
        SparseMatrix _actionWeights; // Action function weights
    };

    // History sample for delayed updates
    struct HistorySample {
        std::vector<IntBuffer> _inputCs;
        IntBuffer _hiddenCsPrev;
        FloatBuffer _hiddenValues;
        
        float _reward;
    };

private:
    Int3 _hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int _historySize;

    IntBuffer _hiddenCs; // Hidden states

    FloatBuffer _hiddenValues; // Hidden value function output buffer

    std::vector<std::shared_ptr<HistorySample>> _historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const IntBuffer*> &inputCsPrev,
        const IntBuffer* hiddenCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Actor* a,
        const std::vector<const IntBuffer*> &inputCs
    ) {
        a->forward(pos, rng, inputCs);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        Actor* a,
        const std::vector<const IntBuffer*> &inputCsPrev,
        const IntBuffer* hiddenCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g
    ) {
        a->learn(pos, rng, inputCsPrev, hiddenCsPrev, hiddenValuesPrev, q, g);
    }

public:
    float _alpha; // Value learning rate
    float _beta; // Action learning rate
    float _gamma; // Discount factor

    int _historyIters; // Sample iters

    // Defaults
    Actor()
    :
    _alpha(0.01f),
    _beta(0.01f),
    _gamma(0.99f),
    _historyIters(16)
    {}

    Actor(
        const Actor &other
    ) {
        *this = other;
    }

    const Actor &operator=(
        const Actor &other
    );

    // Initialized randomly
    void initRandom(
        ComputeSystem &cs,
        const Int3 &hiddenSize,
        int historyCapacity,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs
    );

    // Step (get actions and update)
    void step(
        ComputeSystem &cs,
        const std::vector<const IntBuffer*> &inputCs,
        const IntBuffer* hiddenCsPrev,
        float reward,
        bool learnEnabled
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get hidden state/output/actions
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }

    // Get the value weights for a visible layer
    const SparseMatrix &getValueWeights(
        int i // Index of layer
    ) {
        return _visibleLayers[i]._valueWeights;
    }

    // Get the action weights for a visible layer
    const SparseMatrix &getActionWeights(
        int i // Index of layer
    ) {
        return _visibleLayers[i]._actionWeights;
    }
};
} // namespace ogmaneo
