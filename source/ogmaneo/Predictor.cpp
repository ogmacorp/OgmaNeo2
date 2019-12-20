// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

void Predictor::init(
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

    // Counts
    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCounts, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "pCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        vl._weights.initLocalRF(cs, vld._size, _hiddenSize, vld._radius, -0.01f, 0.01f, rng);

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        vl._visibleCsPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));

        cs.getQueue().enqueueFillBuffer(vl._visibleCsPrev, static_cast<cl_int>(0), 0, numVisibleColumns * sizeof(cl_int));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0), 0, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "pForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "pInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "pLearn");
}

void Predictor::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &visibleCs,
    const cl::Buffer &hiddenTargetCs,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    if (learnEnabled) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnKernel.setArg(argIndex++, vl._visibleCsPrev);
            _learnKernel.setArg(argIndex++, _hiddenActivations);
            _learnKernel.setArg(argIndex++, hiddenTargetCs);
            _learnKernel.setArg(argIndex++, _hiddenCounts);
            _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
            _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
            _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
        }
    }

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Compute feed stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenActivations);
        _forwardKernel.setArg(argIndex++, vl._weights._nonZeroValues);
        _forwardKernel.setArg(argIndex++, vl._weights._rowRanges);
        _forwardKernel.setArg(argIndex++, vl._weights._columnIndices);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));

        // Copy to prevs
        cs.getQueue().enqueueCopyBuffer(visibleCs[vli], vl._visibleCsPrev, 0, 0, numVisibleColumns * sizeof(cl_int));
    }

    // Inhibit
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void Predictor::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(cl_float));

    writeBufferToStream(cs, os, _hiddenCs, numHiddenColumns * sizeof(cl_int));
    writeBufferToStream(cs, os, _hiddenActivations, numHidden * sizeof(cl_int));

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<const char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.writeToStream(cs, os);

        writeBufferToStream(cs, os, vl._visibleCsPrev, numVisibleColumns * sizeof(cl_int));
    }
}

void Predictor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));

    readBufferFromStream(cs, is, _hiddenCs, numHiddenColumns * sizeof(cl_int));
    readBufferFromStream(cs, is, _hiddenActivations, numHidden * sizeof(cl_int));

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.readFromStream(cs, is);

        readBufferFromStream(cs, is, vl._visibleCsPrev, numVisibleColumns * sizeof(cl_int));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "pForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "pInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "pLearn");

    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCounts, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "pCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}