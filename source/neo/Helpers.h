// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ThreadPool.h"

#include <random>
#include <future>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>

namespace ogmaneo {
    class ComputeSystem;
    
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
    //!@}

    //!@{
    /*!
    \brief Kernel executors, 1-3D
    */
    void runKernel1(ComputeSystem &cs, const std::function<void(int, std::mt19937 &rng)> &func, int size, std::mt19937 &rng, int batchSize);
    void runKernel2(ComputeSystem &cs, const std::function<void(const Int2 &, std::mt19937 &rng)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize);
    void runKernel3(ComputeSystem &cs, const std::function<void(const Int3 &, std::mt19937 &rng)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize);
    //!@}

    //!@{
    /*!
    \brief Some useful default kernels
    */
    void fillInt(int pos, std::mt19937 &rng, IntBuffer* buffer, int fillValue);
    void fillFloat(int pos, std::mt19937 &rng, FloatBuffer* buffer, float fillValue);
    void copyInt(int pos, std::mt19937 &rng, const IntBuffer* src, IntBuffer* dst);
    void copyFloat(int pos, std::mt19937 &rng, const FloatBuffer* src, FloatBuffer* dst);
    //!@}
        
    //!@{
    /*!
    \brief Bounds check functions
    */
    inline bool inBounds0(const Int2 &position, const Int2 &upperBound) {
        return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
    }

    inline bool inBounds(const Int2 &position, const Int2 &lowerBound, const Int2 &upperBound) {
        return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
    }
    //!@}

    //!@{
    /*!
    \brief Projection functions
    */
    inline Int2 project(const Int2 &position, const Float2 &toScalars) {
        return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
    }

    inline Int2 projectf(const Float2 &position, const Float2 &toScalars) {
        return Int2(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
    }
    //!@}

    //!@{
    /*!
    \brief High dimensional addressing functions (Nd to 1d)
    */
    inline int address2(const Int2 &pos, int dim) {
        return pos.x + pos.y * dim;
    }

    inline int address3(const Int3 &pos, const Int2 &dims) {
        return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
    }

    inline int address4(const Int4 &pos, const Int3 &dims) {
        int dxy = dims.x * dims.y;
        int dxyz = dxy * dims.z;

        return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
    }
    //!@}

    //!@{
    /*!
    \brief Vector of shared pointers to vector of pointers
    */
    std::vector<IntBuffer*> get(const std::vector<std::shared_ptr<IntBuffer>> &v);
    std::vector<FloatBuffer*> get(const std::vector<std::shared_ptr<FloatBuffer>> &v);
    std::vector<const IntBuffer*> constGet(const std::vector<std::shared_ptr<IntBuffer>> &v);
    std::vector<const FloatBuffer*> constGet(const std::vector<std::shared_ptr<FloatBuffer>> &v);
    //!@}

    /*!
    \brief Sigmoid
    */
    float sigmoid(float x);
}