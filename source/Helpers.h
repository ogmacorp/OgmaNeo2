// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ThreadPool.h"

#include "SparseMatrix.h"

#include <random>
#include <future>
#include <vector>
#include <array>
#include <functional>
#include <ostream>
#include <istream>
#include <assert.h>

namespace ogmaneo {
class ComputeSystem;

// Vector types
template <typename T> 
struct Vec2 {
    T x, y;

    Vec2() {}

    Vec2(
        T X,
        T Y
    )
    : x(X), y(Y)
    {}
};

template <typename T> 
struct Vec3 {
    T x, y, z;
    T pad;

    Vec3()
    {}

    Vec3(
        T X,
        T Y,
        T Z
    )
    : x(X), y(Y), z(Z)
    {}
};

template <typename T> 
struct Vec4 {
    T x, y, z, w;

    Vec4()
    {}

    Vec4(
        T X,
        T Y,
        T Z,
        T W
    )
    : x(X), y(Y), z(Z), w(W)
    {}
};

// Some basic definitions
typedef Vec2<int> Int2;
typedef Vec3<int> Int3;
typedef Vec4<int> Int4;
typedef Vec2<float> Float2;
typedef Vec3<float> Float3;
typedef Vec4<float> Float4;

typedef std::vector<int> IntBuffer;
typedef std::vector<float> FloatBuffer;

// --- Kernel Executors ---

void runKernel1(
    ComputeSystem &cs, // Compute system
    const std::function<void(int, std::mt19937 &rng)> &func, // Kernel function
    int size, // Execution extent size
    std::mt19937 &rng, // Generator
    int batchSize // Batch size
);

void runKernel2(
    ComputeSystem &cs, // Compute system
    const std::function<void(const Int2 &, std::mt19937 &rng)> &func, // Kernel function
    const Int2 &size, // Execution extent size
    std::mt19937 &rng, // Generator
    const Int2 &batchSize // Batch size
);

// Run 3D kernel
void runKernel3(
    ComputeSystem &cs, // Compute system
    const std::function<void(const Int3 &, std::mt19937 &rng)> &func, // Kernel function
    const Int3 &size, // Execution extent size
    std::mt19937 &rng, // Generator
    const Int3 &batchSize // Batch size
);

// --- Basic Kernels ---

// Copy kernel
void fillInt(
    int pos, // Position
    std::mt19937 &rng, // Generator
    IntBuffer* buffer, // Fill buffer
    int fillValue // Value to fill
);

// Copy kernel
void fillFloat(
    int pos, // Position
    std::mt19937 &rng, // Generator
    FloatBuffer* buffer, // Fill buffer
    float fillValue // Value to fill
);

// Copy kernel
void copyInt(
    int pos, // Position
    std::mt19937 &rng, // Generator
    const IntBuffer* src, // Source buffer
    IntBuffer* dst // Destination buffer
);

// Copy kernel
void copyFloat(
    int pos, // Position
    std::mt19937 &rng, // Generator
    const FloatBuffer* src, // Source buffer
    FloatBuffer* dst // Destination buffer
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

inline Int2 projectf(
    const Float2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2(pos.x * toScalars.x + 0.5f, pos.y * toScalars.y + 0.5f);
}

// --- Addressing ---

// Row-major versions
inline int address2R(
    const Int2 &pos, // Position
    int dim // Dimension to ravel with
) {
    return pos.x + pos.y * dim;
}

inline int address3R(
    const Int3 &pos, // Position
    const Int2 &dims // Dimensions to ravel with
) {
    return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
}

inline int address4R(
    const Int4 &pos, // Position
    const Int3 &dims // Dimensions to ravel with
) {
    int dxy = dims.x * dims.y;
    int dxyz = dxy * dims.z;

    return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
}

// Column-major versions
inline int address2C(
    const Int2 &pos, // Position
    const Int2 &dims // Dimensions to ravel with
) {
    return pos.y + pos.x * dims.y;
}

inline int address3C(
    const Int3 &pos, // Position
    const Int3 &dims // Dimensions to ravel with
) {
    return pos.z + pos.y * dims.z + pos.x * dims.z * dims.y;
}

inline int address4C(
    const Int4 &pos, // Position
    const Int4 &dims // Dimensions to ravel with
) {
    return pos.w + pos.z * dims.w + pos.y * dims.w * dims.z + pos.x * dims.w * dims.z * dims.y;
}

// --- Getters ---

std::vector<IntBuffer*> get(
    std::vector<std::shared_ptr<IntBuffer>> &v
);

std::vector<FloatBuffer*> get(
    std::vector<std::shared_ptr<FloatBuffer>> &v
);

std::vector<const IntBuffer*> constGet(
    const std::vector<std::shared_ptr<IntBuffer>> &v
);

std::vector<const FloatBuffer*> constGet(
    const std::vector<std::shared_ptr<FloatBuffer>> &v
);

std::vector<IntBuffer*> get(
    std::vector<IntBuffer> &v
);

std::vector<FloatBuffer*> get(
    std::vector<FloatBuffer> &v
);

std::vector<const IntBuffer*> constGet(
    const std::vector<IntBuffer> &v
);

std::vector<const FloatBuffer*> constGet(
    const std::vector<FloatBuffer> &v
);

// --- Noninearities ---

inline float sigmoid(
    float x
) {
    return 1.0f / (1.0f + std::exp(-x));
}

// --- Serialization ---

template <class T>
void writeBufferToStream(
    std::ostream &os, // Stream
    const std::vector<T>* buf // Buffer to write
) {
    int size = buf->size();

    os.write(reinterpret_cast<const char*>(&size), sizeof(int));
    os.write(reinterpret_cast<const char*>(buf->data()), size * sizeof(T));
}

template <class T>
void readBufferFromStream(
    std::istream &is, // Stream
    std::vector<T>* buf // Buffer to write
) {
    int size;

    is.read(reinterpret_cast<char*>(&size), sizeof(int));

    if (buf->size() != size)
        buf->resize(size);

    is.read(reinterpret_cast<char*>(buf->data()), size * sizeof(T));
}

// --- Sparse Matrix Generation ---

// Sparse matrix init
void initSMLocalRF(
    const Int3 &inSize, // Size of input field
    const Int3 &outSize, // Size of output field
    int radius, // Radius of output onto input
    SparseMatrix &mat // Matrix to fill
);

// --- Sparse Matrix Serialization ---

void writeSMToStream(
    std::ostream &os, // Stream to write to
    const SparseMatrix &mat // Matrix to write to stream
);

void readSMFromStream(
    std::istream &is, // Stream to read from
    SparseMatrix &mat // Matrix to read from stream
);
} // namespace ogmaneo