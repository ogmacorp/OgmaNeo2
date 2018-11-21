// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <random>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>

namespace ogmaneo {
    /*!
    \brief 2D vector type
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
    \brief 3D vector type
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
    \brief 4D vector type
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
    \brief Common type definitions
    */
    typedef Vec2<int> Int2;
    typedef Vec3<int> Int3;
    typedef Vec4<int> Int4;
    typedef Vec2<float> Float2;
    typedef Vec3<float> Float3;
    typedef Vec4<float> Float4;
    //!@}

    //!@{
    /*!
    \brief Two main buffer types
    */
    typedef std::vector<int> IntBuffer;
    typedef std::vector<float> FloatBuffer;

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
    typedef std::array<IntBuffer, 2> IntDoubleBuffer;
    typedef std::array<FloatBuffer, 2> FloatDoubleBuffer;
    //!@}

    //!@{
    /*!
    \brief Default kernel executors
    */
    class KernelWorkItem1 : public WorkItem {
    public:
        std::function<void(int, std::mt19937 &rng)> _func;
        int _pos;
        int _batchSize;

        std::mt19937 _rng;

        KernelWorkItem1()
        {}

        void run() override;
    };

    class KernelWorkItem2 : public WorkItem {
    public:
        std::function<void(const Int2 &, std::mt19937 &rng)> _func;
        Int2 _pos;
        Int2 _batchSize;

        std::mt19937 _rng;

        KernelWorkItem2()
        {}

        void run() override;
    };

    class KernelWorkItem3 : public WorkItem {
    public:
        std::function<void(const Int3 &, std::mt19937 &rng)> _func;
        Int3 _pos;
        Int3 _batchSize;

        std::mt19937 _rng;

        KernelWorkItem3()
        {}

        void run() override;
    };
    //!@}

    //!@{
    /*!
    \brief Kernel executors
    */
    void runKernel1(ComputeSystem &cs, const std::function<void(int, std::mt19937 &rng)> &func, int size, std::mt19937 &rng, int batchSize);
    void runKernel2(ComputeSystem &cs, const std::function<void(const Int2 &, std::mt19937 &rng)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize);
    void runKernel3(ComputeSystem &cs, const std::function<void(const Int3 &, std::mt19937 &rng)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize);
    //!@}

    //!@{
    /*!
    \brief Some useful default kernels
    */
    void fillInt(int pos, std::mt19937 &rng, IntBuffer* buffer, int fillValue) {
        (*buffer)[pos] = fillValue;
    }

    void fillFloat(int pos, std::mt19937 &rng, FloatBuffer* buffer, float fillValue) {
        (*buffer)[pos] = fillValue;
    }

    void copyInt(int pos, std::mt19937 &rng, IntBuffer* src, IntBuffer* dst) {
        (*dst)[pos] = (*src)[pos];
    }

    void copyFloat(int pos, std::mt19937 &rng, FloatBuffer* src, FloatBuffer* dst) {
        (*dst)[pos] = (*src)[pos];
    }
    //!@}
        
    //!@{
    /*!
    \brief Misc functions
    */
    bool inBounds0(const Int2 &position, const Int2 &upperBound) {
    return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
    }

    bool inBounds(const Int2 &position, const Int2 &lowerBound, const Int2 &upperBound) {
        return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
    }

    Int2 project(const Int2 &position, const Float2 &toScalars) {
        return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
    }

    Int2 projectf(const Float2 &position, const Float2 &toScalars) {
        return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
    }

    int address2(const Int2 &pos, int dim) {
        return pos.x + pos.y * dim;
    }

    int address3(const Int3 &pos, const Int2 &dims) {
        return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
    }

    int address4(const Int4 &pos, const Int3 &dims) {
        int dxy = dims.x * dims.y;
        int dxyz = dxy * dims.z;

        return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
    }
    //!@}
}