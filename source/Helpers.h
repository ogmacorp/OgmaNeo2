// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"
#include "ComputeProgram.h"

#include <random>
#include <array>
#include <assert.h>

namespace ogmaneo {
template <typename T> 
struct Vec2 {
    T x, y;

    Vec2()
    {}

    Vec2(T X, T Y)
    : x(X), y(Y)
    {}
};

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

template <typename T> 
struct Vec4 {
    T x, y, z, w;

    Vec4()
    {}

    Vec4(T X, T Y, T Z, T W)
    : x(X),y(Y), z(Z), w(W)
    {}
};

typedef Vec2<cl_int> Int2;
typedef Vec3<cl_int> Int3;
typedef Vec4<cl_int> Int4;
typedef Vec2<cl_float> Float2;
typedef Vec3<cl_float> Float3;
typedef Vec4<cl_float> Float4;

enum BufferType {
    _front = 0,
    _back = 1
};

typedef std::array<cl::Buffer, 2> DoubleBuffer;

DoubleBuffer createDoubleBuffer(
    ComputeSystem &cs,
    cl_int size
);

// --- Bounds ---

// Bounds check from (0, 0) to upperBound
inline bool inBounds0(
    const Int2 &pos, // Position
    const Int2 &upperBound // Bottom-right corner
) {
    return pos.x >= 0 && pos.x < upperBound.x && pos.y >= 0 && pos.y < upperBound.y;
}

// Bounds check in range
inline bool inBounds(
    const Int2 &pos, // Position
    const Int2 &lowerBound, // Top-left corner
    const Int2 &upperBound // Bottom-right corner
) {
    return pos.x >= lowerBound.x && pos.x < upperBound.x && pos.y >= lowerBound.y && pos.y < upperBound.y;
}

// --- Projections ---

inline Int2 project(
    const Int2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2(pos.x * toScalars.x + 0.5f, pos.y * toScalars.y + 0.5f);
}

inline Int2 project(
    const Float2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2(pos.x * toScalars.x + 0.5f, pos.y * toScalars.y + 0.5f);
}

// Row-major accessors
inline int address2(
    const Int2 &pos, // Position
    const Int2 &dims // Dimensions to ravel with
) {
    return pos.y + pos.x * dims.y;
}

inline int address3(
    const Int3 &pos, // Position
    const Int3 &dims // Dimensions to ravel with
) {
    return pos.z + pos.y * dims.z + pos.x * dims.z * dims.y;
}

inline int address4(
    const Int4 &pos, // Position
    const Int4 &dims // Dimensions to ravel with
) {
    return pos.w + pos.z * dims.w + pos.y * dims.w * dims.z + pos.x * dims.w * dims.z * dims.y;
}

void writeBufferToStream(ComputeSystem &cs, std::ostream &os, cl::Buffer &buf, int size);
void readBufferFromStream(ComputeSystem &cs, std::istream &is, cl::Buffer &buf, int size);
} // namespace ogmaneo