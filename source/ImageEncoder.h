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
// Encodes images (dense -> CSDR)
class ImageEncoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 _size; // Size of input

        int _radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        : 
        _size({ 4, 4, 16 }),
        _radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        SparseMatrix _weights; // Weight matrix

        FloatBuffer _visibleActivations; // For reconstruction

        IntBuffer _visibleCounts; // For reconstruction
    };

private:
    Int3 _hiddenSize; // Hidden layer size

    IntBuffer _hiddenCs; // Hidden state

    FloatBuffer _hiddenRates; // Rates
    
    std::vector<VisibleLayer> _visibleLayers; // Layers
    std::vector<VisibleLayerDesc> _visibleLayerDescs; // Descs

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        std::mt19937 &rng,
        const std::vector<const FloatBuffer*> &inputActivations,
        bool learnEnabled
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
        const std::vector<const FloatBuffer*> &inputActivations,
        bool learnEnabled
    ) {
        sc->forward(pos, rng, inputActivations, learnEnabled);
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
    float _alpha; // Learning rate
    float _beta; // Learning rate decay
    float _gamma; // SOM falloff

    // Initialize defaults
    ImageEncoder()
    :
    _alpha(1.0f),
    _beta(0.99f),
    _gamma(0.2f)
    {}

    // Create a randomly initialized image encoder
    void initRandom(
        ComputeSystem &cs, // Compute system
        const Int3 &hiddenSize, // Size of the hidden layer
        const std::vector<VisibleLayerDesc> &visibleLayerDescs // Descs
    );

    // Step the image encoder
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const FloatBuffer*> &inputActivations, // Input state (activations)
        bool learnEnabled = true // Whether to learn
    );

    // Reconstruct (reverse) mapping
    void reconstruct(
        ComputeSystem &cs, // Compute system
        const IntBuffer* hiddenCs // Hidden states to reconstruct
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );
    
    // Get the number of visible (input) layers
    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of the visible layer
    ) const {
        return _visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of the visible layer
    ) const {
        return _visibleLayerDescs[i];
    }

    // Get the hidden state
    const IntBuffer &getHiddenCs() const {
        return _hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return _hiddenSize;
    }

    // Get a visible layer's feed weights
    const SparseMatrix &getWeights(
        int i // Index of the visible layer
    ) const {
        return _visibleLayers[i]._weights;
    }
};
} // namespace ogmaneo
