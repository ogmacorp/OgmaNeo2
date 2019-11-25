// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
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
enum InputType {
    _none,
    _act
};

class Hierarchy {
public:
    struct FirstLayerDesc {
        Int3 _hiddenSize;

        cl_int _scRadius;
        cl_int _aRadius;

        int _temporalHorizon;

        int _historyCapacity;

        FirstLayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _scRadius(2),
        _aRadius(2),
        _temporalHorizon(2),
        _historyCapacity(8)
        {}
    };

    struct HigherLayerDesc {
        Int3 _hiddenSize;

        cl_int _scRadius;
        cl_int _pRadius;

        int _ticksPerUpdate;

        int _temporalHorizon;

        HigherLayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _scRadius(2),
        _pRadius(2),
        _ticksPerUpdate(2),
        _temporalHorizon(2)
        {}
    };
private:
    std::vector<SparseCoder> _scLayers;
    std::vector<std::vector<Predictor>> _pLayers; // Prediction layers for all but bottom of hierarchy
    std::vector<std::unique_ptr<Actor>> _aLayers; // Action layers at bottom of hierarchy

    std::vector<std::vector<cl::Buffer>> _histories;
    std::vector<std::vector<int>> _historySizes;

    std::vector<unsigned char> _updates;

    std::vector<int> _ticks;
    std::vector<int> _ticksPerUpdate;

    std::vector<Int3> _inputSizes;

public:
    void init(
        ComputeSystem &cs,
        ComputeProgram &prog,
        const std::vector<Int3> &inputSizes,
        const std::vector<InputType> &inputTypes,
        const FirstLayerDesc &firstLayerDesc,
        const std::vector<HigherLayerDesc> &higherLayerDescs,
        std::mt19937 &rng
    );

    void step(
        ComputeSystem &cs,
        const std::vector<cl::Buffer> &inputCs,
        std::mt19937 &rng,
        float reward,
        bool learn = true
    );

    void writeToStream(
        ComputeSystem &cs,
        std::ostream &os
    );

    void readFromStream(
        ComputeSystem &cs,
        ComputeProgram &prog,
        std::istream &is
    );

    int getNumLayers() const {
        return _scLayers.size();
    }

    const cl::Buffer &getActionCs(
        int i
    ) const {
        assert(_aLayers[i] != nullptr);

        return _aLayers[i]->getHiddenCs();
    }

    bool getUpdate(
        int l
    ) const {
        return _updates[l];
    }

    int getTicks(
        int l
    ) const {
        return _ticks[l];
    }

    int getTicksPerUpdate(
        int l
    ) const {
        return _ticksPerUpdate[l];
    }

    const std::vector<Int3> &getInputSizes() const {
        return _inputSizes;
    }

    SparseCoder &getSCLayer(
        int l
    ) {
        return _scLayers[l];
    }

    const SparseCoder &getSCLayer(
        int l
    ) const {
        return _scLayers[l];
    }

    std::vector<Predictor> &getPLayers(
        int l
    ) {
        int pl = l - 1;

        return _pLayers[pl];
    }

    const std::vector<Predictor> &getPLayers(
        int l
    ) const {
        int pl = l - 1;

        return _pLayers[pl];
    }

    std::vector<std::unique_ptr<Actor>> &getALayers() {
        return _aLayers;
    }

    const std::vector<std::unique_ptr<Actor>> &getALayers() const {
        return _aLayers;
    }
};
} // namespace ogmaneo
