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

#include <array>
#include <random>
#include <assert.h>

namespace ogmaneo {
    /*!
    \brief 2D vector type.
    */
    template <typename T> 
    struct Vec2 {
        T x, y;

        Vec2()
        {}

        Vec2(T X, T Y)
        : x(X), y(Y)
        {}
    };

    /*!
    \brief 3D vector type.
    */
    template <typename T> 
    struct Vec3 {
        T x, y, z;
        T pad;

        Vec3()
        {}

        Vec3(T X, T Y, T Z)
        : x(X), y(Y), z(Z)
        {}
    };

    /*!
    \brief 4D vector type.
    */
    template <typename T> 
    struct Vec4 {
        T x, y, z, w;

        Vec4()
        {}

        Vec4(T X, T Y, T Z, T W)
        : x(X), y(Y), z(Z), w(W)
        {}
    };

    //!@{
    /*!
    \brief Common type definitions.
    */
    typedef Vec2<cl_int> Int2;
    typedef Vec3<cl_int> Int3;
    typedef Vec4<cl_int> Int4;
    typedef Vec2<cl_float> Float2;
    typedef Vec3<cl_float> Float3;
    typedef Vec4<cl_float> Float4;
    //!@}

    /*!
    \brief Buffer types (can be used as indices)
    */
    enum BufferType {
        _front = 0, _back = 1
    };

    /*!
    \brief Double buffer types
    */
    typedef std::array<cl::Buffer, 2> DoubleBuffer;

    /*!
    \brief Double buffer creation helpers
    */
    DoubleBuffer createDoubleBuffer(ComputeSystem &cs, cl_int size);
}