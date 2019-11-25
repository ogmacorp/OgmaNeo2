// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseMatrix.h"

namespace ogmaneo {
class SparseCoder {
public:
    struct VisibleLayerDesc {
        Int3 _size;

        cl_int _radius;

        VisibleLayerDesc()
        :
        _size(8, 8, 16),
        _radius(2)
        {}
    };

    struct VisibleLayer {
        SparseMatrix _weights;
    };

private:
    Int3 _hiddenSize;

    cl::Buffer _hiddenCs;

    cl::Buffer _hiddenActivations;

    std::vector<VisibleLayer> _visibleLayers;
    std::vector<VisibleLayerDesc> _visibleLayerDescs;

    cl::Kernel _forwardKernel;
    cl::Kernel _inhibitKernel;
    cl::Kernel _learnKernel;

public:
    cl_float _alpha;

    SparseCoder()
    :
    _alpha(0.1f)
    {}

    void init(
        ComputeSystem &cs,
        ComputeProgram &prog,
        Int3 hiddenSize,
        const std::vector<VisibleLayerDesc> &visibleLayerDescs,
        std::mt19937 &rng
    );

    void step(
        ComputeSystem &cs,
        const std::vector<cl::Buffer> &visibleCs,
        std::mt19937 &rng,
        bool learnEnabled
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

    int getNumVisibleLayers() const {
        return _visibleLayers.size();
    }

    const VisibleLayer &getVisibleLayer(
        int index
    ) const {
        return _visibleLayers[index];
    }

    const VisibleLayerDesc &getVisibleLayerDesc(
        int index
    ) const {
        return _visibleLayerDescs[index];
    }

    const cl::Buffer &getHiddenCs() const {
        return _hiddenCs;
    }

    Int3 getHiddenSize() const {
        return _hiddenSize;
    }
};
} // namespace ogmaneo
