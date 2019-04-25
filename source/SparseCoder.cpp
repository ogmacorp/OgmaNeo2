// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::init(
    ComputeSystem &cs,
    ComputeProgram &prog,
    Int3 hiddenSize,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng
) {
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._weights.initLocalRF(cs, vld._size, _hiddenSize, vld._radius, -0.001f, 0.0f, rng);

        vl._weights.initT(cs);

        vl._visibleErrors = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    _refractoryTimers = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_refractoryTimers, static_cast<cl_int>(0), 0, numHidden * sizeof(cl_int));
 
    // Hidden activations
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _backwardKernel = cl::Kernel(prog.getProgram(), "scBackward");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
}

void SparseCoder::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &visibleCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Forward
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenActivations);
        _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
        _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
        _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }

    // Inhibit
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _refractoryTimers);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    if (learnEnabled) {
        // Learn
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            {
                int argIndex = 0;

                _backwardKernel.setArg(argIndex++, visibleCs[vli]);
                _backwardKernel.setArg(argIndex++, _hiddenCs);
                _backwardKernel.setArg(argIndex++, vl._visibleErrors);
                _backwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
                _backwardKernel.setArg(argIndex++, vl._weights._nonZeroValueIndices);
                _backwardKernel.setArg(argIndex++, vl._weights._columnRanges);
                _backwardKernel.setArg(argIndex++, vl._weights._rowIndices);
                _backwardKernel.setArg(argIndex++, vld._size);
                _backwardKernel.setArg(argIndex++, _hiddenSize);
                _backwardKernel.setArg(argIndex++, _alpha);

                cs.getQueue().enqueueNDRangeKernel(_backwardKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y, vld._size.z));
            }

            {
                int argIndex = 0;

                _learnKernel.setArg(argIndex++, vl._visibleErrors);
                _learnKernel.setArg(argIndex++, _hiddenCs);
                _learnKernel.setArg(argIndex++, _refractoryTimers);
                _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
                _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
                _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
                _learnKernel.setArg(argIndex++, _hiddenSize);
                _learnKernel.setArg(argIndex++, _refractoryTicks);

                cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
            }
        }
    }
}

void SparseCoder::writeToStream(
    ComputeSystem &cs,
    std::ostream &os
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    cs.getQueue().enqueueReadBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
    os.write(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.writeToStream(cs, os);
    }
}

void SparseCoder::readFromStream(
    ComputeSystem &cs,
    ComputeProgram &prog,
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._weights.readFromStream(cs, is);

        vl._visibleErrors = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _backwardKernel = cl::Kernel(prog.getProgram(), "scBackward");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
}