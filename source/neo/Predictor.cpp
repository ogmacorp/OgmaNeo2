// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

void Predictor::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    cl_int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    cl::Kernel initWeightsKernel = cl::Kernel(prog.getProgram(), "pInitWeights");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._visibleSize.x * vld._visibleSize.y;

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._visibleSize.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._visibleSize.y) / static_cast<float>(_hiddenSize.y)
        };

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._visibleSize.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));

        {
            std::uniform_int_distribution<int> seedDist(0, 99999);

            int argIndex = 0;

            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, cl_uint2{ static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng)) });

            cs.getQueue().enqueueNDRangeKernel(initWeightsKernel, cl::NullRange, cl::NDRange(weightsSize));
        }

        vl._visibleCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));

        cs.getQueue().enqueueFillBuffer(vl._visibleCs, static_cast<cl_int>(0), 0, numVisibleColumns * sizeof(cl_int));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));
 
    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "pForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "pInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "pLearn");
}

void Predictor::activate(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    int numColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Compute feed stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Copy visible activation
        cs.getQueue().enqueueCopyBuffer(visibleCs[vli], vl._visibleCs,
            0, 0, vld._visibleSize.x * vld._visibleSize.y * sizeof(cl_int));

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenActivations);
        _forwardKernel.setArg(argIndex++, vl._weights);
        _forwardKernel.setArg(argIndex++, vld._visibleSize);
        _forwardKernel.setArg(argIndex++, _hiddenSize);
        _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
        _forwardKernel.setArg(argIndex++, vld._radius);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }

    // Activate
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void Predictor::learn(ComputeSystem &cs, const cl::Buffer &targetCs) {
    // Learn feed
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _learnKernel.setArg(argIndex++, vl._visibleCs);
        _learnKernel.setArg(argIndex++, _hiddenActivations);
        _learnKernel.setArg(argIndex++, targetCs);
        _learnKernel.setArg(argIndex++, vl._weights);
        _learnKernel.setArg(argIndex++, vld._visibleSize);
        _learnKernel.setArg(argIndex++, _hiddenSize);
        _learnKernel.setArg(argIndex++, vl._hiddenToVisible);
        _learnKernel.setArg(argIndex++, vld._radius);
        _learnKernel.setArg(argIndex++, _alpha);

        cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }
}