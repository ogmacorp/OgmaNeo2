// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"
#include "Actor.h"

#include <memory>

namespace ogmaneo {
enum InputType {
    _none,
    _act
};

class Hierarchy {
public:
    struct LayerDesc {
        Int3 _hiddenSize;

        cl_int _scRadius;
        cl_int _aRadius;

        int _ticksPerUpdate;

        int _temporalHorizon;

        int _historyCapacity;

        LayerDesc()
        :
        _hiddenSize(4, 4, 16),
        _scRadius(2),
        _aRadius(2),
        _ticksPerUpdate(2),
        _temporalHorizon(2),
        _historyCapacity(8)
        {}
    };
private:
    std::vector<SparseCoder> _scLayers;
    std::vector<std::vector<std::unique_ptr<Actor>>> _aLayers;

    std::vector<std::vector<cl::Buffer>> _histories;
    std::vector<std::vector<int>> _historySizes;

    std::vector<unsigned char> _updates;

    std::vector<int> _ticks;
    std::vector<int> _ticksPerUpdate;

    std::vector<float> _rewards;
    std::vector<float> _rewardCounts;

    std::vector<Int3> _inputSizes;

public:
    void init(
        ComputeSystem &cs,
        ComputeProgram &prog,
        const std::vector<Int3> &inputSizes,
        const std::vector<InputType> &inputTypes,
        const std::vector<LayerDesc> &layerDescs,
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
        assert(_aLayers.front()[i] != nullptr);

        return _aLayers.front()[i]->getHiddenCs();
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

    std::vector<std::unique_ptr<Actor>> &getALayers(
        int l
    ) {
        return _aLayers[l];
    }

    const std::vector<std::unique_ptr<Actor>> &getALayers(
        int l
    ) const {
        return _aLayers[l];
    }
};
} // namespace ogmaneo
