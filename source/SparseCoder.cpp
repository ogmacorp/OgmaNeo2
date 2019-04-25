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
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
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

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        vl._visibleToHidden = Float2(static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y));

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        vl._reverseRadii = Int2(static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1));

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

        vl._visibleActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Hidden activations
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _backwardPartialKernel = cl::Kernel(prog.getProgram(), "scBackwardPartial");
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

        cs.getQueue().enqueueFillBuffer(vl._visibleActivations, static_cast<cl_float>(0.0f), 0, vld._size.x * vld._size.y * vld._size.z * sizeof(cl_float)); 
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
            _forwardKernel.setArg(argIndex++, vld._size);
            _forwardKernel.setArg(argIndex++, _hiddenSize);
            _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
            _forwardKernel.setArg(argIndex++, vld._radius);

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

        // Backward partially
        if (it < _explainIters - 1) {
            for (int vli = 0; vli < _visibleLayers.size(); vli++) {
                VisibleLayer &vl = _visibleLayers[vli];
                VisibleLayerDesc &vld = _visibleLayerDescs[vli];

                int argIndex = 0;

                _backwardPartialKernel.setArg(argIndex++, visibleCs[vli]);
                _backwardPartialKernel.setArg(argIndex++, _hiddenCs);
                _backwardPartialKernel.setArg(argIndex++, vl._visibleActivations);
                _backwardPartialKernel.setArg(argIndex++, vl._weights);
                _backwardPartialKernel.setArg(argIndex++, vld._size);
                _backwardPartialKernel.setArg(argIndex++, _hiddenSize);
                _backwardPartialKernel.setArg(argIndex++, vl._visibleToHidden);
                _backwardPartialKernel.setArg(argIndex++, vl._hiddenToVisible);
                _backwardPartialKernel.setArg(argIndex++, vld._radius);
                _backwardPartialKernel.setArg(argIndex++, vl._reverseRadii);

                cs.getQueue().enqueueNDRangeKernel(_backwardPartialKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
            }
        }
    }
}

void SparseCoder::learn(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs) {
    // Learn
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _backwardKernel.setArg(argIndex++, _hiddenCs);
            _backwardKernel.setArg(argIndex++, vl._visibleActivations);
            _backwardKernel.setArg(argIndex++, vl._weights);
            _backwardKernel.setArg(argIndex++, vld._size);
            _backwardKernel.setArg(argIndex++, _hiddenSize);
            _backwardKernel.setArg(argIndex++, vl._visibleToHidden);
            _backwardKernel.setArg(argIndex++, vl._hiddenToVisible);
            _backwardKernel.setArg(argIndex++, vld._radius);
            _backwardKernel.setArg(argIndex++, vl._reverseRadii);

            cs.getQueue().enqueueNDRangeKernel(_backwardKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y, vld._size.z));
        }

        {
            int argIndex = 0;

            _learnKernel.setArg(argIndex++, visibleCs[vli]);
            _learnKernel.setArg(argIndex++, vl._visibleActivations);
            _learnKernel.setArg(argIndex++, _hiddenCs);
            _learnKernel.setArg(argIndex++, vl._weights);
            _learnKernel.setArg(argIndex++, vld._size);
            _learnKernel.setArg(argIndex++, _hiddenSize);
            _learnKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnKernel.setArg(argIndex++, vld._radius);
            _learnKernel.setArg(argIndex++, _alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }
    }
}

void SparseCoder::writeToStream(ComputeSystem &cs, std::ostream &os) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    os.write(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    os.write(reinterpret_cast<char*>(&_explainIters), sizeof(cl_int));

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

        os.write(reinterpret_cast<char*>(&vl._visibleToHidden), sizeof(Float2));
        os.write(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));
        os.write(reinterpret_cast<char*>(&vl._reverseRadii), sizeof(Int2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        cs.getQueue().enqueueReadBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());
        os.write(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
    }
}

void SparseCoder::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
    is.read(reinterpret_cast<char*>(&_hiddenSize), sizeof(Int3));

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    is.read(reinterpret_cast<char*>(&_alpha), sizeof(cl_float));
    is.read(reinterpret_cast<char*>(&_explainIters), sizeof(cl_int));

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

        is.read(reinterpret_cast<char*>(&vl._visibleToHidden), sizeof(Float2));
        is.read(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));
        is.read(reinterpret_cast<char*>(&vl._reverseRadii), sizeof(Int2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        is.read(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());

        vl._visibleActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _backwardPartialKernel = cl::Kernel(prog.getProgram(), "scBackwardPartial");
    _backwardKernel = cl::Kernel(prog.getProgram(), "scBackward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
}