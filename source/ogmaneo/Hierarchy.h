// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"
#include "Predictor.h"
#include "Actor.h"

#include <memory>

namespace ogmaneo {
// Type of hierarchy input layer
enum InputType {
    none = 0,
    prediction = 1,
    action = 2
};

// State of hierarchy
struct State {
    std::vector<IntBuffer> hiddenCs;
    std::vector<IntBuffer> hiddenCsPrev;
    std::vector<std::vector<std::vector<IntBuffer>>> predInputCsPrev;
    std::vector<std::vector<std::vector<IntBuffer>>> predInputCsPrevPrev;
    std::vector<std::vector<IntBuffer>> predHiddenCs;

    std::vector<std::vector<IntBuffer>> histories;

    std::vector<char> updates;
    std::vector<int> ticks;
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int pRadius; // Prediction radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)

        int temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        // If there is an actor (only valid for first layer)
        int aRadius;
        int historyCapacity;

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        pRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(4),
        aRadius(2),
        historyCapacity(32)
        {}
    };
private:
    // Layers
    std::vector<SparseCoder> scLayers;
    std::vector<std::vector<std::unique_ptr<Predictor>>> pLayers;
    std::vector<std::unique_ptr<Actor>> aLayers;

    // Histories
    std::vector<std::vector<std::shared_ptr<IntBuffer>>> histories;
    std::vector<std::vector<int>> historySizes;

    // Per-layer values
    std::vector<char> updates;

    std::vector<int> ticks;
    std::vector<int> ticksPerUpdate;

    // Input dimensions
    std::vector<Int3> inputSizes;

public:
    // Default
    Hierarchy() {}

    // Copy
    Hierarchy(
        const Hierarchy &other // Hierarchy to copy from
    ) {
        *this = other;
    }

    // Assignment
    const Hierarchy &operator=(
        const Hierarchy &other // Hierarchy to assign from
    );
    
    // Create a randomly initialized hierarchy
    void initRandom(
        ComputeSystem &cs, // Compute system
        const std::vector<Int3> &inputSizes, // Sizes of input layers
        const std::vector<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
        const std::vector<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        ComputeSystem &cs, // Compute system
        const std::vector<const IntBuffer*> &inputCs, // Inputs to remember
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f, // Optional reward for actor layers
        bool mimic = false
    );

    // State get
    void getState(
        State &state
    ) const;

    // State set
    void setState(
        const State &state
    );

    // Write to stream
    void writeToStream(
        std::ostream &os // Stream to write to
    ) const;

    // Read from stream
    void readFromStream(
        std::istream &is // Stream to read from
    );

    // Get the number of layers (scLayers)
    int getNumLayers() const {
        return scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCs();

        return pLayers.front()[i]->getHiddenCs();
    }

    // Whether this layer received on update this timestep
    bool getUpdate(
        int l // Layer index
    ) const {
        return updates[l];
    }

    // Get current layer ticks, relative to previous layer
    int getTicks(
        int l // Layer Index
    ) const {
        return ticks[l];
    }

    // Get layer ticks per update, relative to previous layer
    int getTicksPerUpdate(
        int l // Layer Index
    ) const {
        return ticksPerUpdate[l];
    }

    // Get input sizes
    const std::vector<Int3> &getInputSizes() const {
        return inputSizes;
    }

    // Retrieve a sparse coding layer
    SparseCoder &getSCLayer(
        int l // Layer index
    ) {
        return scLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const SparseCoder &getSCLayer(
        int l // Layer index
    ) const {
        return scLayers[l];
    }

    // Retrieve predictor layer(s)
    std::vector<std::unique_ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const std::vector<std::unique_ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) const {
        return pLayers[l];
    }

    // Retrieve predictor layer(s)
    std::vector<std::unique_ptr<Actor>> &getALayers() {
        return aLayers;
    }

    // Retrieve predictor layer(s), const version
    const std::vector<std::unique_ptr<Actor>> &getALayers() const {
        return aLayers;
    }
};
} // namespace ogmaneo
