// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

void Actor::init(
    ComputeSystem &cs,
    ComputeProgram &prog,
    Int3 hiddenSize,
    int historyCapacity,
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

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "aCount");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._weights.initLocalRF(cs, vld._size, _hiddenSize, vld._radius, -0.01f, 0.01f, rng);

        int argIndex = 0;

        countKernel.setArg(argIndex++, vl._weights._rowRanges);
        countKernel.setArg(argIndex++, _hiddenCounts);
        countKernel.setArg(argIndex++, vld._size);
        countKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(countKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));
    _hiddenActivationsPartial = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_float));

    // History samples
    _historySize = 0;
    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i]._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            _historySamples[i]._visibleCs[vli] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
        }

        _historySamples[i]._hiddenCsPrev = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _forwardPartialKernel = cl::Kernel(prog.getProgram(), "aForwardPartial");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &visibleCs,
    const cl::Buffer &hiddenCsPrev,
    std::mt19937 &rng,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Compute feed stimulus
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

    // Activate
    {
        std::uniform_int_distribution<int> seedDist(0, 9999999);

        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Add sample
    if (_historySize == _historySamples.size()) {
        // Circular buffer swap
        HistorySample temp = _historySamples.front();

        for (int i = 0; i < _historySamples.size() - 1; i++) {
            _historySamples[i] = _historySamples[i + 1];
        }

        _historySamples.back() = temp;
    }

    if (_historySize < _historySamples.size())
        _historySize++;
    
    {
        HistorySample &s = _historySamples[_historySize - 1];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            // Copy visible Cs
            cs.getQueue().enqueueCopyBuffer(visibleCs[vli], s._visibleCs[vli],
                0, 0, numVisibleColumns * sizeof(cl_int));
        }

        cs.getQueue().enqueueCopyBuffer(hiddenCsPrev, s._hiddenCsPrev, 0, 0, numHiddenColumns * sizeof(cl_int));

        s._reward = reward;
    }

    // Learn
    if (learnEnabled && _historySize > 1) {
        const HistorySample &sPrev = _historySamples[0];
        const HistorySample &s = _historySamples[1];

        cl_float q = 0.0f;
        cl_float g = 1.0f;

        for (int t = 1; t < _historySize; t++) {
            q += _historySamples[t]._reward * g;
            g *= _gamma;
        }

        // Initialize stimulus to 0
        cs.getQueue().enqueueFillBuffer(_hiddenActivationsPartial, static_cast<cl_float>(0.0f), 0, numHiddenColumns * sizeof(cl_float));

        // Compute feed stimulus
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _forwardPartialKernel.setArg(argIndex++, sPrev._visibleCs[vli]);
            _forwardPartialKernel.setArg(argIndex++, s._hiddenCsPrev);
            _forwardPartialKernel.setArg(argIndex++, _hiddenActivationsPartial);
            _forwardPartialKernel.setArg(argIndex++, vl._weights._nonZeroValues);
            _forwardPartialKernel.setArg(argIndex++, vl._weights._rowRanges);
            _forwardPartialKernel.setArg(argIndex++, vl._weights._columnIndices);
            _forwardPartialKernel.setArg(argIndex++, vld._size);
            _forwardPartialKernel.setArg(argIndex++, _hiddenSize);

            cs.getQueue().enqueueNDRangeKernel(_forwardPartialKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnKernel.setArg(argIndex++, sPrev._visibleCs[vli]);
            _learnKernel.setArg(argIndex++, _hiddenActivations);
            _learnKernel.setArg(argIndex++, _hiddenActivationsPartial);
            _learnKernel.setArg(argIndex++, _hiddenCs);
            _learnKernel.setArg(argIndex++, s._hiddenCsPrev);
            _learnKernel.setArg(argIndex++, _hiddenCounts);
            _learnKernel.setArg(argIndex++, vl._weights._nonZeroValues);
            _learnKernel.setArg(argIndex++, vl._weights._rowRanges);
            _learnKernel.setArg(argIndex++, vl._weights._columnIndices);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, _alpha);
            _learnKernel.setArg(argIndex++, g);
            _learnKernel.setArg(argIndex++, q);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }
    }
}

void Actor::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<const char*>(&_gamma), sizeof(cl_float));

    writeBufferToStream(cs, os, _hiddenCs, numHiddenColumns * sizeof(cl_int));

    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<const char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.writeToStream(cs, os);
    }

    int historyCapacity = _historySamples.size();

    os.write(reinterpret_cast<const char*>(&historyCapacity), sizeof(int));
    os.write(reinterpret_cast<const char*>(&_historySize), sizeof(int));

    for (int i = 0; i < _historySamples.size(); i++) {
        HistorySample &s = _historySamples[i];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            writeBufferToStream(cs, os, s._visibleCs[vli], numVisibleColumns * sizeof(cl_int));
        }

        writeBufferToStream(cs, os, s._hiddenCsPrev, numHiddenColumns * sizeof(cl_int));

        os.write(reinterpret_cast<const char*>(&s._reward), sizeof(cl_float));
    }
}

void Actor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));

    readBufferFromStream(cs, is, _hiddenCs, numHiddenColumns * sizeof(cl_int));

    // Stimulus
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));
    _hiddenActivationsPartial = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_float));

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    _visibleLayers.resize(numVisibleLayers);
    _visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        vl._weights.readFromStream(cs, is);
    }

    int historyCapacity;

    is.read(reinterpret_cast<char*>(&historyCapacity), sizeof(int));
    is.read(reinterpret_cast<char*>(&_historySize), sizeof(int));

    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        HistorySample &s = _historySamples[i];

        s._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            readBufferFromStream(cs, is, s._visibleCs[vli], numVisibleColumns * sizeof(cl_int));
        }

        readBufferFromStream(cs, is, s._hiddenCsPrev, numHiddenColumns * sizeof(cl_int));

        is.read(reinterpret_cast<char*>(&_historySamples[i]._reward), sizeof(cl_float));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _forwardPartialKernel = cl::Kernel(prog.getProgram(), "aForwardPartial");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");

    _hiddenCounts = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCounts, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));

    cl::Kernel countKernel = cl::Kernel(prog.getProgram(), "aCount");

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