// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <assert.h>

namespace ogmaneo {
class ComputeProgram {
private:
    cl::Program _program;

    bool loadFromString(
        ComputeSystem &cs,
        const std::string &prog
    );

public:
    bool loadFromFile(
        ComputeSystem &cs,
        const std::string &name
    );

    cl::Program &getProgram() {
        return _program;
    }
};
} // namespace ogmaneo