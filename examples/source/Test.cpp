// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <ogmaneo/system/ComputeSystem.h>
#include <ogmaneo/neo/SparseCoder.h>
#include <ogmaneo/neo/Hierarchy.h>

#include "matplotlibcpp.h"

#include <CL/cl.h>

#include <iostream>

namespace plt = matplotlibcpp;

#define TEST_WAVY
//#define TEST_ACTOR

#if defined(TEST_WAVY)
int main() {
    std::mt19937 rng(time(nullptr));

    ogmaneo::ComputeSystem cs;
    cs.create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::ComputeProgram prog;
    prog.loadFromFile(cs, "../../resources/neoKernels.cl");

    int inputSize = 64;

    std::vector<ogmaneo::Hierarchy::LayerDesc> lds(4);

    for (int l = 0; l < lds.size(); l++) {
        lds[l]._hiddenSize = cl_int3{ 5, 5, 16 };
    }

    ogmaneo::Hierarchy h;

    h.createRandom(cs, prog, { cl_int3{ 1, 1, inputSize } }, { ogmaneo::_predict }, lds, rng);

    cl::Buffer inputBuf = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, 1 * sizeof(cl_int));
    cl::Buffer topFeedBack = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(topFeedBack, static_cast<cl_int>(0), 0, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    int iters = 5000;

    // Iterate
    for (int it = 0; it < iters; it++) {
        float value = std::sin(it * 0.02f * 2.0f * 3.14159f) * 0.25f + std::sin(it * 0.05f * 2.0f * 3.14159f) * 0.15f + (std::fmod(it * 0.01f, 1.25f) * 2.0f - 1.0f) * 0.2f;
        
        int index = static_cast<int>((value * 0.5f + 0.5f) * (inputSize - 1) + 0.5f);

        std::vector<cl_int> inputs(1);
        inputs[0] = index;

        // Create buffer
        cs.getQueue().enqueueWriteBuffer(inputBuf, CL_TRUE, 0, 1 * sizeof(cl_int), inputs.data());

        h.step(cs, { inputBuf }, topFeedBack, true);

        // Print prediction
        std::vector<cl_int> preds(1);

        cs.getQueue().enqueueReadBuffer(h.getPredictionCs(0), CL_TRUE, 0, 1 * sizeof(cl_int), preds.data());

        int maxIndex = preds[0];

        std::vector<cl_int> actBuf(lds[0]._hiddenSize.x * lds[0]._hiddenSize.y);

        cs.getQueue().enqueueReadBuffer(h.getSCLayer(0).getHiddenCs(), CL_TRUE, 0, actBuf.size() * sizeof(cl_int), actBuf.data());

        float nextValue = maxIndex / static_cast<float>(inputSize - 1) * 2.0f - 1.0f;

        //std::cout << value << " " << nextValue << std::endl;

        if (it % 10 == 0)
            std::cout << "Iter " << it << std::endl;
    }

    std::vector<float> yvals;
    std::vector<float> yvals2;

    for (int it2 = 0; it2 < 500; it2++) {
        int it = iters + it2;

        float value = std::sin(it * 0.02f * 2.0f * 3.14159f) * 0.25f + std::sin(it * 0.05f * 2.0f * 3.14159f) * 0.15f + (std::fmod(it * 0.01f, 1.25f) * 2.0f - 1.0f) * 0.2f;

        h.step(cs, { h.getPredictionCs(0) }, topFeedBack, true);

        // Print prediction
        std::vector<cl_int> preds(1);

        cs.getQueue().enqueueReadBuffer(h.getPredictionCs(0), CL_TRUE, 0, 1 * sizeof(cl_int), preds.data());

        int maxIndex = preds[0];

        std::vector<cl_int> actBuf(lds[0]._hiddenSize.x * lds[0]._hiddenSize.y);

        cs.getQueue().enqueueReadBuffer(h.getSCLayer(0).getHiddenCs(), CL_TRUE, 0, actBuf.size() * sizeof(cl_int), actBuf.data());

        float nextValue = maxIndex / static_cast<float>(inputSize - 1) * 2.0f - 1.0f;

        std::cout << value << " " << nextValue << std::endl;

        yvals.push_back(nextValue);
        yvals2.push_back(value);

        if (it % 10 == 0)
            std::cout << "Iter " << it << std::endl;
    }

    plt::plot(yvals);
    plt::plot(yvals2);
    plt::show();

    return 0;
}
#elif defined(TEST_ACTOR)
int main() {
    std::mt19937 rng(time(nullptr));

    ogmaneo::ComputeSystem cs;
    cs.create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::ComputeProgram prog;
    prog.loadFromFile(cs, "../../resources/neoKernels.cl");

    int inputSize = 32;

    std::vector<ogmaneo::Hierarchy::LayerDesc> lds(4);

    for (int l = 0; l < lds.size(); l++) {
        lds[l]._hiddenSize = cl_int3{ 3, 3, 32 };
    }

    ogmaneo::Hierarchy h;

    h.createRandom(cs, prog, { cl_int3{ 1, 1, inputSize } }, std::vector<ogmaneo::InputType>{ ogmaneo::_predict }, lds, rng);

    cl::Buffer inputBuf = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, 1 * sizeof(cl_int));
    cl::Buffer topFeedBack = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(topFeedBack, static_cast<cl_int>(0), 0, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    std::vector<float> yvals;

    int iters = 2000;

    // Iterate
    for (int it = 0; it < iters; it++) {
        float value = std::sin(it * 0.1f);
        
        int index = static_cast<int>((value * 0.5f + 0.5f) * (inputSize - 1) + 0.5f);

        std::vector<cl_int> inputs(1);
        inputs[0] = index;

        // Create buffer
        cs.getQueue().enqueueWriteBuffer(inputBuf, CL_TRUE, 0, 1 * sizeof(cl_int), inputs.data());

        h.step(cs, { inputBuf }, topFeedBack, true);

        // Print prediction
        std::vector<cl_int> preds(1);

        cs.getQueue().enqueueReadBuffer(h.getPredictionCs(0), CL_TRUE, 0, 1 * sizeof(cl_int), preds.data());

        int maxIndex = preds[0];

        std::vector<cl_int> actBuf(lds[0]._hiddenSize.x * lds[0]._hiddenSize.y);

        cs.getQueue().enqueueReadBuffer(h.getSCLayer(0).getHiddenCs(), CL_TRUE, 0, actBuf.size() * sizeof(cl_int), actBuf.data());

        float nextValue = maxIndex / static_cast<float>(inputSize - 1) * 2.0f - 1.0f;

        std::cout << value << " " << nextValue << std::endl;

        if (it % 10 == 0)
            std::cout << "Iter " << it << std::endl;
    }

    return 0;
}
#endif