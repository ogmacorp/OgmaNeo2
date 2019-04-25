// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

using namespace ogmaneo;

DoubleBuffer ogmaneo::createDoubleBuffer(ComputeSystem &cs, cl_int size) {
    DoubleBuffer db;

    db[_front] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, size);
    db[_back] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, size);

    return db;
}