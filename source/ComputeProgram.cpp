// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ComputeProgram.h"

#include <fstream>
#include <iostream>
#include <numeric>

using namespace ogmaneo;

bool ComputeProgram::loadFromFile(
    ComputeSystem &cs,
    const std::string &name
) {
    std::ifstream fromFile(name);

    if (!fromFile.is_open()) {
#ifdef SYS_DEBUG
        std::cerr << "Could not open file " << name << "!" << std::endl;
#endif
        return false;
    }

    std::string kernel = "";

    while (!fromFile.eof() && fromFile.good()) {
        std::string line;

        std::getline(fromFile, line);

        kernel += line + "\n";
    }

    return loadFromString(cs, kernel);
}

bool ComputeProgram::loadFromString(
    ComputeSystem &cs,
    const std::string& prog
) {
    _program = cl::Program(cs.getContext(), prog);

    if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS) {
#ifdef SYS_DEBUG
        std::cerr << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << std::endl;
#endif
        return false;
    }

    return true;
}
