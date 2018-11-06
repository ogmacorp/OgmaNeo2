// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

void ImageEncoder::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    cl::Kernel initWeightsKernel = cl::Kernel(prog.getProgram(), "imInitWeights");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

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

        vl._visibleAs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));

        cs.getQueue().enqueueFillBuffer(vl._visibleAs, static_cast<cl_float>(0.0f), 0, numVisible * sizeof(cl_float));
    }

    // Hidden Cs
    _hiddenCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHiddenColumns * sizeof(cl_int));

    cs.getQueue().enqueueFillBuffer(_hiddenCs, static_cast<cl_int>(0), 0, numHiddenColumns * sizeof(cl_int));
 
    // Hidden activations
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "imForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "imInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "imLearn");
}

void ImageEncoder::activate(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleAs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Forward
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _forwardKernel.setArg(argIndex++, visibleAs[vli]);
        _forwardKernel.setArg(argIndex++, vl._visibleAs);
        _forwardKernel.setArg(argIndex++, _hiddenActivations);
        _forwardKernel.setArg(argIndex++, vl._weights);
        _forwardKernel.setArg(argIndex++, vld._size);
        _forwardKernel.setArg(argIndex++, _hiddenSize);
        _forwardKernel.setArg(argIndex++, vl._hiddenToVisible);
        _forwardKernel.setArg(argIndex++, vld._radius);

        cs.getQueue().enqueueNDRangeKernel(_forwardKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y, _hiddenSize.z));

        // Copy
        cs.getQueue().enqueueCopyBuffer(visibleAs[vli], vl._visibleAs, 0, 0, vld._size.x * vld._size.y * vld._size.z * sizeof(cl_float));
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

void ImageEncoder::learn(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleAs) {
    // Learn
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _learnKernel.setArg(argIndex++, visibleAs[vli]);
        _learnKernel.setArg(argIndex++, vl._visibleAs);
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

void ImageEncoder::writeToStream(ComputeSystem &cs, std::ostream &os) {
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

        os.write(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        cs.getQueue().enqueueReadBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());
        os.write(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));

        std::vector<cl_float> visibleAs(numVisible);
        cs.getQueue().enqueueReadBuffer(vl._visibleAs, CL_TRUE, 0, numVisible * sizeof(cl_float), visibleAs.data());
        os.write(reinterpret_cast<char*>(visibleAs.data()), numVisible * sizeof(cl_float));
    }
}

void ImageEncoder::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
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

        is.read(reinterpret_cast<char*>(&vl._hiddenToVisible), sizeof(Float2));

        cl_int diam = vld._radius * 2 + 1;

        cl_int numWeightsPerHidden = diam * diam * vld._size.z;

        cl_int weightsSize = numHidden * numWeightsPerHidden;

        std::vector<cl_float> weights(weightsSize);
        is.read(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(cl_float));
        vl._weights = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, weightsSize * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._weights, CL_TRUE, 0, weightsSize * sizeof(cl_float), weights.data());

        std::vector<cl_float> visibleAs(numVisible);
        is.read(reinterpret_cast<char*>(visibleAs.data()), numVisible * sizeof(cl_float));
        vl._visibleAs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisible * sizeof(cl_float));
        cs.getQueue().enqueueWriteBuffer(vl._visibleAs, CL_TRUE, 0, numVisible * sizeof(cl_float), visibleAs.data());
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "imForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "imInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "imLearn");
}