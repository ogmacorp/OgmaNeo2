// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

#include <iostream>
using namespace ogmaneo;

void Actor::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    Int3 hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    cl::Kernel initWeightsKernel = cl::Kernel(prog.getProgram(), "aInitWeights");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));

        {
            std::uniform_int_distribution<int> seedDist(0, 99999);

            int argIndex = 0;

            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, Vec2<cl_uint>(static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng))));

            cs.getQueue().enqueueNDRangeKernel(initWeightsKernel, cl::NullRange, cl::NDRange(weightsSize));
        }
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

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

        _historySamples[i]._hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}

void Actor::step(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs, std::mt19937 &rng, float reward, bool learn) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations[_front], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Compute feed stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleCs[vli]);
        _forwardKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _forwardKernel.setArg(argIndex++, vl._weights);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);
        _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
        _forwardKernel.setArg(argIndex++, vld._radius);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
    }

    // Activate
    {
        std::uniform_int_distribution<int> seedDist(0, 99999);

        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, _epsilon);
        _inhibitKernel.setArg(argIndex++, Vec2<cl_uint>(static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng))));

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

        cs.getQueue().enqueueCopyBuffer(_hiddenCs, s._hiddenCs, 0, 0, numHiddenColumns * sizeof(cl_int));

        s._reward = reward;
    }

    // Learn
    if (learn && _historySize > 1) {
        const HistorySample &sPrev = _historySamples[0];

        cl_float q = 0.0f;

        for (int t = _historySize - 1; t >= 1; t--)
            q += _historySamples[t]._reward * std::pow(_gamma, t - 1);

        // Initialize stimulus to 0
        cs.getQueue().enqueueFillBuffer(_hiddenActivations[_back], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

        // Compute feed stimulus
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _forwardKernel.setArg(argIndex++, sPrev._visibleCs[vli]);
            _forwardKernel.setArg(argIndex++, _hiddenActivations[_back]);
            _forwardKernel.setArg(argIndex++, vl._weights);
            _forwardKernel.setArg(argIndex++, vld._size);
            _forwardKernel.setArg(argIndex++, _hiddenSize);
            _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
            _forwardKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));
        }

        cl_float g = std::pow(_gamma, _historySize - 1);

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnKernel.setArg(argIndex++, sPrev._visibleCs[vli]);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_front]);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_back]);
            _learnKernel.setArg(argIndex++, _hiddenCs);
            _learnKernel.setArg(argIndex++, sPrev._hiddenCs);
            _learnKernel.setArg(argIndex++, vl._weights);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnKernel.setArg(argIndex++, vld._radius);
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

    os.write(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_epsilon), sizeof(cl_float));

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

        os.write(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        cs.getQueue().enqueueReadBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());
        os.write(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
    }

    int historyCapacity = _historySamples.size();

    os.write(reinterpret_cast<char*>(&historyCapacity), sizeof(int));
    os.write(reinterpret_cast<char*>(&_historySize), sizeof(int));

    for (int i = 0; i < _historySize; i++) {
        HistorySample &s = _historySamples[i];

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;
            int numVisible = numVisibleColumns * vld._size.z;

            std::vector<cl_int> visibleCs(numVisibleColumns);
            cs.getQueue().enqueueReadBuffer(s._visibleCs[vli], CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
            os.write(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
        }

        std::vector<cl_int> hiddenCs(numHiddenColumns);
        cs.getQueue().enqueueReadBuffer(s._hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
        os.write(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

        os.write(reinterpret_cast<char*>(&s._reward), sizeof(cl_float));
    }
}

void Actor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_epsilon), sizeof(cl_float));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

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

        is.read(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        is.read(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());
    }

    int historyCapacity, historySize;

    is.read(reinterpret_cast<char*>(&historyCapacity), sizeof(int));
    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    _historySamples.resize(historyCapacity);

    for (int i = 0; i < _historySamples.size(); i++) {
        _historySamples[i]._visibleCs.resize(_visibleLayers.size());

        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int numVisibleColumns = vld._size.x * vld._size.y;

            std::vector<cl_int> visibleCs(numVisibleColumns);
            is.read(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
            _historySamples[i]._visibleCs[vli] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
            cs.getQueue().enqueueWriteBuffer(_historySamples[i]._visibleCs[vli], CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
        }

        std::vector<cl_int> hiddenCs(numHiddenColumns);
        is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
        _historySamples[i]._hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
        cs.getQueue().enqueueWriteBuffer(_historySamples[i]._hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

        is.read(reinterpret_cast<char*>(&_historySamples[i]._reward), sizeof(cl_float));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}