// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
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
    /*!
    \brief Compute system
    Holds OpenCL platform, device, context, and command queue
    */
    class ComputeSystem {
    public:
        /*!
        \brief OpenCL device types
        */
        enum DeviceType {
            _cpu, _gpu, _all
        };

    private:
        //!@{
        /*!
        \brief OpenCL handles
        */
        cl::Platform _platform;
        cl::Device _device;
        cl::Context _context;
        cl::CommandQueue _queue;
        //!@}

    public:
        ComputeSystem() {}

        // Non-copyable
        ComputeSystem(const ComputeSystem &) = delete;
        ComputeSystem &operator=(const ComputeSystem &) = delete;

        /*!
        \brief Create an OpenCL compute system with a given device type.
        Optional: Create from a platform index, device index, and an OpenGL context
        Default: Use the last platform and last device discovered
        */
        bool create(DeviceType type, int platformIndex = -1, int deviceIndex = -1, bool createFromGLContext = false);

        /*!
        \brief Get underlying OpenCL platform
        */
        cl::Platform &getPlatform() {
            return _platform;
        }

        /*!
        \brief Get underlying OpenCL device
        */
        cl::Device &getDevice() {
            return _device;
        }

        /*!
        \brief Get underlying OpenCL context
        */
        cl::Context &getContext() {
            return _context;
        }

        /*!
        \brief Get underlying OpenCL command queue
        */
        cl::CommandQueue &getQueue() {
            return _queue;
        }
    };
}