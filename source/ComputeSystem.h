// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
//#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define SYS_DEBUG

#define SYS_ALLOW_CL_GL_CONTEXT 0

namespace ogmaneo {
    class ComputeSystem {
    public:
        enum DeviceType {
            _cpu,
            _gpu,
            _all
        };

    private:
        cl::Platform _platform;
        cl::Device _device;
        cl::Context _context;
        cl::CommandQueue _queue;

    public:
        ComputeSystem() {}

        // Non-copyable
        ComputeSystem(
            const ComputeSystem &
        ) = delete;
        
        ComputeSystem &operator=(
            const ComputeSystem &
        ) = delete;

        bool init(
            DeviceType type,
            int platformIndex = -1,
            int deviceIndex = -1,
            bool createFromGLContext = false
        );

        cl::Platform &getPlatform() {
            return _platform;
        }

        cl::Device &getDevice() {
            return _device;
        }

        cl::Context &getContext() {
            return _context;
        }

        cl::CommandQueue &getQueue() {
            return _queue;
        }
    };
}