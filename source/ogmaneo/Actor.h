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
        SparseMatrix weights; // Q weights
        SparseMatrix traces; // Q traces

        IntBuffer inputCsPrev;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    IntBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenActivations; // Activations of actions
    FloatBuffer hiddenActivationsPrev; // Activations of actions from the previous timestep

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
        const IntBuffer* hiddenCsPrev,
        float reward
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
        const IntBuffer* hiddenCsPrev,
        float reward
    ) {
        a->learn(pos, rng, hiddenCsPrev, reward);
    }

public:
    float alpha; // Value learning rate
    float gamma; // Discount factor
    float traceDecay; // Trace decay multiplier

    // Defaults
    Actor()
    :
    alpha(0.1f),
    gamma(0.99f),
    traceDecay(0.97f)
    {}

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

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of layer
    ) {
        return visibleLayers[i].weights;
    }
};
} // namespace ogmaneo
