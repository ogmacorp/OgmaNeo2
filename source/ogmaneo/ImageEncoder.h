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
// Sparse coder
class ImageEncoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

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
        SparseMatrix weights; // Weight matrix

        FloatBuffer reconErrors;
        FloatBuffer reconActs;
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenActivations;

    FloatBuffer hiddenResources; // Resources

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> visibleLayers;
    std::vector<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActs
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    void backwardErrors(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        const FloatBuffer* inputActs,
        int vli
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActs
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const std::vector<const FloatBuffer*> &inputActs
    ) {
        sc->forward(pos, rng, inputActs);
    }

    static void backwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const IntBuffer* hiddenCs,
        int vli
    ) {
        sc->backward(pos, rng, hiddenCs, vli);
    }

    static void backwardErrorsKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const IntBuffer* hiddenCs,
        const FloatBuffer* inputActs,
        int vli
    ) {
        sc->backwardErrors(pos, rng, hiddenCs, inputActs, vli);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const std::vector<const FloatBuffer*> &inputActs
    ) {
        sc->learn(pos, rng, inputActs);
    }

public:
    float alpha; // Resource depletion rate
    float beta; // Error reduction
    float gamma; // Gas falloff

    // Defaults
    ImageEncoder()
    :
    alpha(0.01f),
    beta(0.5f),
    gamma(0.01f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputActs, // Input states
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        ComputeSystem &cs, // Compute system
        const IntBuffer* hiddenCs
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace ogmaneo
