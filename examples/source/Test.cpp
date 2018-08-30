// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <ogmaneo/system/ComputeSystem.h>
#include <ogmaneo/neo/SparseCoder.h>
#include <ogmaneo/neo/ImageEncoder.h>
#include <ogmaneo/neo/Hierarchy.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "matplotlibcpp.h"

#include <CL/cl.h>

#include <iostream>

namespace plt = matplotlibcpp;

int address4(cl_int4 pos, cl_int3 dims) {
    int dxy = dims.x * dims.y;
    int dxyz = dxy * dims.z;

    return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

#define TEST_SC
//#define TEST_WAVY
//#define TEST_ACTOR

#if defined(TEST_SC)
int main() {
    std::mt19937 rng(time(nullptr));

    ogmaneo::ComputeSystem cs;
    cs.create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::ComputeProgram prog;
    prog.loadFromFile(cs, "../../resources/neoKernels.cl");

    int patchWidth = 16;
    int patchHeight = 16;

    int hiddenWidth = 8;
    int hiddenHeight = 8;
    int hiddenColumnSize = 32;

    ogmaneo::ImageEncoder sc;
    sc._explainIters = 4;
    sc._alpha = 0.05f;

    std::vector<ogmaneo::ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0]._visibleSize = cl_int3{ patchWidth, patchHeight, 3 };
    vlds[0]._radius = 6;

    sc.createRandom(cs, prog, cl_int3{ hiddenWidth, hiddenHeight, hiddenColumnSize }, vlds, rng);

    // Load image
    sf::Image img;
    img.loadFromFile("cat.png");

    std::uniform_int_distribution<int> startDistX(0, img.getSize().x - patchWidth - 1);
    std::uniform_int_distribution<int> startDistY(0, img.getSize().y - patchHeight - 1);

    cl::Buffer patchBuf = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, patchWidth * patchHeight * 3 * sizeof(float));

    // Iterate
    for (int it = 0; it < 10000; it++) {
        // Select random portion of image
        int startX = startDistX(rng);
        int startY = startDistY(rng);

        std::vector<float> data(patchWidth * patchHeight * 3);

        for (int px = 0; px < patchWidth; px++)
            for (int py = 0; py < patchHeight; py++) {
                sf::Color c = img.getPixel(startX + px, startY + py);

                data[px + py * patchWidth + 0 * patchWidth * patchHeight] = c.r / 255.0f;
                data[px + py * patchWidth + 1 * patchWidth * patchHeight] = c.g / 255.0f;
                data[px + py * patchWidth + 2 * patchWidth * patchHeight] = c.b / 255.0f;
            }

        // Create buffer
        cs.getQueue().enqueueWriteBuffer(patchBuf, CL_TRUE, 0, data.size() * sizeof(float), data.data());

        sc.activate(cs, { patchBuf });

        // std::vector<float> actBuf(hiddenWidth * hiddenHeight);

        // cs.getQueue().enqueueReadBuffer(sc.getHiddenActs(), CL_TRUE, 0, actBuf.size() * sizeof(float), actBuf.data());

        // for (int i = 0; i < actBuf.size(); i++)
        //     std::cout << actBuf[i] << " ";

        // std::cout << std::endl;
        // std::cout << std::endl;

        sc.learn(cs, { patchBuf });

        if (it % 10 == 0)
            std::cout << "Iter " << it << std::endl;
    }

    std::cout << "Saving some weights..." << std::endl;

    int diam = sc.getVisibleLayerDesc(0)._radius * 2 + 1;
    int diam2 = diam * diam;
    int weightsPerUnit = diam2 * sc.getVisibleLayerDesc(0)._visibleSize.z;
    int totalNumWeights = weightsPerUnit * hiddenWidth * hiddenHeight * hiddenColumnSize;

    std::vector<cl_float> weights(totalNumWeights);

    std::cout << "Has " << totalNumWeights << " weights." << std::endl;

    cs.getQueue().enqueueReadBuffer(sc.getWeights(0), CL_TRUE, 0, totalNumWeights * sizeof(cl_float), weights.data());

    for (int colSlice = 0; colSlice < hiddenColumnSize; colSlice++) {
        sf::Image wImg;
        wImg.create(hiddenWidth * diam, hiddenHeight * diam, sf::Color::Black);

        for (int x = 0; x < hiddenWidth; x++)
            for (int y = 0; y < hiddenHeight; y++) {
                for (int dx = 0; dx < diam; dx++)
                    for (int dy = 0; dy < diam; dy++) {
                        int tx = x * diam + dx;
                        int ty = y * diam + dy;

                        int iR = dx + dy * diam + 0 * diam2;
                        int iG = dx + dy * diam + 1 * diam2;
                        int iB = dx + dy * diam + 2 * diam2;

                        float wR = weights[address4(cl_int4{ x, y, colSlice, iR }, sc.getHiddenSize())];
                        float wG = weights[address4(cl_int4{ x, y, colSlice, iG }, sc.getHiddenSize())];
                        float wB = weights[address4(cl_int4{ x, y, colSlice, iB }, sc.getHiddenSize())];

                        sf::Color c;
                        c.r = 255 * std::min(1.0f, std::max(0.0f, wR));
                        c.g = 255 * std::min(1.0f, std::max(0.0f, wG));
                        c.b = 255 * std::min(1.0f, std::max(0.0f, wB));

                        wImg.setPixel(tx, ty, c);
                    }
            }

        wImg.saveToFile("wImg" + std::to_string(colSlice) + ".png");
    }

    return 0;
}
#elif defined(TEST_WAVY)
int main() {
    std::mt19937 rng(time(nullptr));

    ogmaneo::ComputeSystem cs;
    cs.create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::ComputeProgram prog;
    prog.loadFromFile(cs, "../../resources/neoKernels.cl");

    int inputSize = 64;

    std::vector<ogmaneo::Hierarchy::LayerDesc> lds(4);

    for (int l = 0; l < lds.size(); l++) {
        lds[l]._hiddenSize = cl_int3{ 4, 4, 16 };
    }

    ogmaneo::Hierarchy h;

    h.createRandom(cs, prog, { cl_int3{ 1, 1, inputSize } }, { ogmaneo::_predict }, lds, rng);

    cl::Buffer inputBuf = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, 1 * sizeof(cl_int));
    cl::Buffer topFeedBack = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(topFeedBack, static_cast<cl_int>(0), 0, lds.back()._hiddenSize.x * lds.back()._hiddenSize.y * sizeof(cl_int));

    int iters = 9000;

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