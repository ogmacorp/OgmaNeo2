// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <assert.h>

namespace ogmaneo {
    /*!
    \brief Compute program
    Holds OpenCL compute program with their associated kernels
    */
    class ComputeProgram {
    private:
        /*!
        \brief OpenCL program
        */
        cl::Program _program;

        /*!
        \brief Load kernel code from a string
        */
        bool loadFromString(ComputeSystem &cs, const std::string& kernel);

    public:
        /*!
        \brief Load kernel code from a file
        */
        bool loadFromFile(ComputeSystem &cs, const std::string &name);

        /*!
        \brief Get the underlying OpenCL program
        */
        cl::Program &getProgram() {
            return _program;
        }
    };
}