// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

void Actor::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    cl_int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
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

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
        };

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        vl._traces = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));

        {
            std::uniform_int_distribution<int> seedDist(0, 99999);

            int argIndex = 0;

            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, cl_uint2{ static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng)) });

            cs.getQueue().enqueueNDRangeKernel(initWeightsKernel, cl::NullRange, cl::NDRange(weightsSize));
        }

        cs.getQueue().enqueueFillBuffer(vl._traces, static_cast<cl_float>(0.0f), 0, weightsSize * sizeof(cl_float));

        vl._visibleCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));

        cs.getQueue().enqueueFillBuffer(vl._visibleCs, static_cast<cl_int>(0), 0, numVisibleColumns * sizeof(cl_int));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Stimulus
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));

    cs.getQueue().enqueueFillBuffer(_hiddenActivations[_front], static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));
 
    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}

void Actor::step(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs, const cl::Buffer &targetCs, float reward, bool learn) {
    int numColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numColumns * _hiddenSize.z;

    // Buffer swap
    std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);

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
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenCs);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Learn
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        if (learn) {
            int argIndex = 0;

            _learnKernel.setArg(argIndex++, vl._visibleCs);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_front]);
            _learnKernel.setArg(argIndex++, _hiddenActivations[_back]);
            _learnKernel.setArg(argIndex++, _hiddenCs);
            _learnKernel.setArg(argIndex++, targetCs);
            _learnKernel.setArg(argIndex++, vl._weights);
            _learnKernel.setArg(argIndex++, vl._traces);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnKernel.setArg(argIndex++, vld._radius);
            _learnKernel.setArg(argIndex++, _alpha);
            _learnKernel.setArg(argIndex++, _gamma);
            _learnKernel.setArg(argIndex++, _traceDecay);
            _learnKernel.setArg(argIndex++, _tdErrorClip);
            _learnKernel.setArg(argIndex++, reward);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Copy visible activation
        cs.getQueue().enqueueCopyBuffer(visibleCs[vli], vl._visibleCs,
            0, 0, vld._size.x * vld._size.y * sizeof(cl_int));
    }
}

void Actor::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<char*>(&_hiddenSize), sizeof(cl_int3));

    os.write(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_traceDecay), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_tdErrorClip), sizeof(cl_float));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    cs.getQueue().enqueueReadBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());
    os.write(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));

    std::vector<cl_float> hiddenActivations(numHidden);
    cs.getQueue().enqueueReadBuffer(_hiddenActivations[_front], CL_TRUE, 0, numHidden * sizeof(cl_float), hiddenActivations.data());
    os.write(reinterpret_cast<char*>(hiddenActivations.data()), numHidden * sizeof(cl_float));
    
    int numVisibleLayers = _visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        os.write(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        os.write(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(cl_float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        cs.getQueue().enqueueReadBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());
        os.write(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));

        std::vector<cl_float> traces(weightsSize);
        cs.getQueue().enqueueReadBuffer(vl._traces, CL_TRUE, 0, weightsSize * sizeof(cl_float), traces.data());
        os.write(reinterpret_cast<char*>(traces.data()), weightsSize * sizeof(cl_float));

        std::vector<cl_int> visibleCs(numVisibleColumns);
        cs.getQueue().enqueueReadBuffer(vl._visibleCs, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
        os.write(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
    }
}

void Actor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(cl_int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_traceDecay), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_tdErrorClip), sizeof(cl_float));

    std::vector<cl_int> hiddenCs(numHiddenColumns);
    is.read(reinterpret_cast<char*>(hiddenCs.data()), numHiddenColumns * sizeof(cl_int));
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));
    cs.getQueue().enqueueWriteBuffer(_hiddenCs, CL_TRUE, 0, numHiddenColumns * sizeof(cl_int), hiddenCs.data());

    std::vector<cl_float> hiddenActivations(numHidden);
    is.read(reinterpret_cast<char*>(hiddenActivations.data()), numHidden * sizeof(cl_float));
    _hiddenActivations = createDoubleBuffer(cs, numHidden * sizeof(cl_float));
    cs.getQueue().enqueueWriteBuffer(_hiddenActivations[_front], CL_TRUE, 0, numHidden * sizeof(cl_float), hiddenActivations.data());

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

        is.read(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(cl_float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        is.read(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());

        std::vector<cl_float> traces(weightsSize);
        is.read(reinterpret_cast<char*>(traces.data()), weightsSize * sizeof(cl_float));
        vl._traces = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._traces, CL_TRUE, 0, weightsSize * sizeof(cl_float), traces.data());

        std::vector<cl_int> visibleCs(numVisibleColumns);
        is.read(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
        vl._visibleCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
        cs.getQueue().enqueueWriteBuffer(vl._visibleCs, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "aForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "aInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "aLearn");
}