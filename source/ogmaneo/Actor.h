// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
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
        Int3 size; // Visible/input size

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix valueWeights; // Value function weights
        SparseMatrix actionWeights; // Action function weights
    };

    // History sample for delayed updates
    struct HistorySample {
        std::vector<IntBuffer> inputCs;
        IntBuffer hiddenTargetCsPrev;

        FloatBuffer hiddenValuesPrev;
        
        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    IntBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenValues; // Hidden value function output buffer

    std::vector<std::shared_ptr<HistorySample>> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;

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
        const IntBuffer* hiddenTargetCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g,
        bool mimic
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
        const IntBuffer* hiddenTargetCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g,
        bool mimic
    ) {
        a->learn(pos, rng, inputCsPrev, hiddenTargetCsPrev, hiddenValuesPrev, q, g, mimic);
    }

public:
    float alpha; // Value learning rate
    float beta; // Action learning rate
    float gamma; // Discount factor
    int minSteps;
    int historyIters;

    // Defaults
    Actor()
    :
    alpha(0.02f),
    beta(0.02f),
    gamma(0.99f),
    minSteps(8),
    historyIters(8)
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
        const IntBuffer* hiddenTargetCsPrev,
        float reward,
        bool learnEnabled,
        bool mimic
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
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get hidden state/output/actions
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace ogmaneo
