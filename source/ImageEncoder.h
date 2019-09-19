// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
        Int3 _size; // Size of input

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
        SparseMatrix _weights; // Weight matrix

        FloatBuffer _reconActivations;
    };

private:
    Int3 _hiddenSize; // Size of hidden/output layer
    int _lateralRadius;

    FloatBuffer _hiddenStimuli;
    FloatBuffer _hiddenActivations;

    IntBuffer _hiddenCs; // Hidden states
    IntBuffer _hiddenCsTemp; // Temporaries for hidden state iteration
    IntBuffer _hiddenUsages; // Number of times used
    
    SparseMatrix _laterals;

    // Visible layers and associated descriptors
    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActivations
    );

    void inhibit(
        const Int2 &pos,
        std::mt19937 &rng
    );

    void learn(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActivations
    );

    void backward(
        const Int2 &pos,
        std::mt19937 &rng,
        const IntBuffer* hiddenCs,
        int vli
    );

    static void forwardKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const std::vector<const FloatBuffer*> &inputActivations
    ) {
        sc->forward(pos, rng, inputActivations);
    }

    static void inhibitKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc
    ) {
        sc->inhibit(pos, rng);
    }

    static void learnKernel(
        const Int2 &pos,
        std::mt19937 &rng,
        ImageEncoder* sc,
        const std::vector<const FloatBuffer*> &inputActivations
    ) {
        sc->learn(pos, rng, inputActivations);
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

public:
    int _explainIters; // Explaining-away iterations

    // Defaults
    ImageEncoder()
    :
    _explainIters(3)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Hidden/output size
        int lateralRadius,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputActivations, // Input activations
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        ComputeSystem &cs, // Compute system
        const IntBuffer* hiddenCs // Hidden state to reconstruct
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
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }

    // Get the weights for a visible layer
    const SparseMatrix &getWeights(
        int i // Index of visible layer
    ) const {
        return _visibleLayers[i]._weights;
    }
};
} // namespace ogmaneo
