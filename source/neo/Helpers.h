// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <random>
#include <assert.h>

namespace ogmaneo {
    /*!
    \brief Buffer types (can be used as indices)
    */
    enum BufferType {
        _front = 0, _back = 1
    };

    //!@{
    /*!
    \brief Double buffer types
    */
    typedef std::array<cl::Buffer, 2> DoubleBuffer;
    //!@}

    //!@{
    /*!
    \brief Double buffer creation helpers
    */
    DoubleBuffer createDoubleBuffer(ComputeSystem &cs, cl_int size);
    //!@}
}