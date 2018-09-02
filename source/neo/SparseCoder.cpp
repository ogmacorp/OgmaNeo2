// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    cl_int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    cl::Kernel initWeightsKernel = cl::Kernel(prog.getProgram(), "scInitWeights");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._visibleSize.x * vld._visibleSize.y;
        int numVisible = numVisibleColumns * vld._visibleSize.z;

        vl._visibleToHidden = cl_float2{ static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._visibleSize.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._visibleSize.y)
        };

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._visibleSize.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._visibleSize.y) / static_cast<float>(_hiddenSize.y)
        };

        vl._reverseRadii = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
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

        vl._visibleActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Hidden activations
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _backwardKernel = cl::Kernel(prog.getProgram(), "scBackward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
}

void SparseCoder::activate(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Initialize visibleActivations to 0
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueFillBuffer(vl._visibleActivations, static_cast<cl_float>(0.0f), 0, vld._visibleSize.x * vld._visibleSize.y * vld._visibleSize.z * sizeof(cl_float)); 
    }

    for (int it = 0; it < _explainIters; it++) {
        // Forward
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _forwardKernel.setArg(argIndex++, visibleCs[vli]);
            _forwardKernel.setArg(argIndex++, vl._visibleActivations);
            _forwardKernel.setArg(argIndex++, _hiddenActivations);
            _forwardKernel.setArg(argIndex++, vl._weights);
            _forwardKernel.setArg(argIndex++, vld._visibleSize);
            _forwardKernel.setArg(argIndex++, _hiddenSize);
            _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
            _forwardKernel.setArg(argIndex++, vld._radius);
            _forwardKernel.setArg(argIndex++, std::pow(0.5f, it)); // Search scalar

            cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
        }

         // Inhibit
        {
           int argIndex = 0;

            _inhibitKernel.setArg(argIndex++, _hiddenActivations);
            _inhibitKernel.setArg(argIndex++, _hiddenCs);
            _inhibitKernel.setArg(argIndex++, _hiddenSize);

            cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Backward
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _backwardKernel.setArg(argIndex++, _hiddenCs);
            _backwardKernel.setArg(argIndex++, vl._visibleActivations);
            _backwardKernel.setArg(argIndex++, vl._weights);
            _backwardKernel.setArg(argIndex++, vld._visibleSize);
            _backwardKernel.setArg(argIndex++, _hiddenSize);
            _backwardKernel.setArg(argIndex++, vl._visibleToHidden);
            _backwardKernel.setArg(argIndex++, vl._hiddenToVisible);
            _backwardKernel.setArg(argIndex++, vld._radius);
            _backwardKernel.setArg(argIndex++, vl._reverseRadii);

            cs.getQueue().enqueueNDRangeKernel(_backwardKernel, cl::NullRange, cl::NDRange(vld._visibleSize.x, vld._visibleSize.y, vld._visibleSize.z));
        }
    }
}

void SparseCoder::learn(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    // Learn
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _learnKernel.setArg(argIndex++, visibleCs[vli]);
        _learnKernel.setArg(argIndex++, vl._visibleActivations);
        _learnKernel.setArg(argIndex++, _hiddenCs);
        _learnKernel.setArg(argIndex++, vl._weights);
        _learnKernel.setArg(argIndex++, vld._visibleSize);
        _learnKernel.setArg(argIndex++, _hiddenSize);
        _learnKernel.setArg(argIndex++, vl._hiddenToVisible);
        _learnKernel.setArg(argIndex++, vld._radius);
        _learnKernel.setArg(argIndex++, _alpha);

        cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}