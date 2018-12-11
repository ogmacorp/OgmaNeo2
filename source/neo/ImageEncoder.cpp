// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace ogmaneo;

<<<<<<< HEAD
// Kernels
void ImageEncoder::init(int pos, std::mt19937 &rng, int vli) {
    // Initialize weights into uniform range
	std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);
=======
void ImageEncoder::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void ImageEncoder::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const FloatBuffer*> &inputActivations) {
    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    // Running max data
    int maxIndex = 0;
    float maxValue = -999999.0f;

    // For each hidden cell
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        // Partial sum cache value
        int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

<<<<<<< HEAD
        // Accumulator
        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);
=======
        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

            // Lower corner
            Int2 fieldLowerBound(visiblePositionCenter.x - vld._radius, visiblePositionCenter.y - vld._radius);

            // Additional addressing dimensions
            int diam = vld._radius * 2 + 1;
            int diam2 = diam * diam;

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(std::max(0, fieldLowerBound.x), std::max(0, fieldLowerBound.y));
            Int2 iterUpperBound(std::min(vld._size.x - 1, visiblePositionCenter.x + vld._radius), std::min(vld._size.y - 1, visiblePositionCenter.y + vld._radius));

            for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
                for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                    for (int vc = 0; vc < vld._size.z; vc++) {
                        Int3 visiblePosition(x, y, vc);

                        int visibleIndex = address3(visiblePosition, Int2(vld._size.x, vld._size.y));

                        float visibleActivation = (*inputActivations[vli])[visibleIndex];
                        float visibleActivationPrev = vl._visibleActivationsPrev[visibleIndex];

<<<<<<< HEAD
                        // Complete the partial address with final value needed
                        int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + vc * diam2;
=======
            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, Vec2<cl_uint>(static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng))));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

                        // Rule is: sum += max(0, weight - prevActivation), found empirically to be better than truncated weight * (1.0 - prevActivation) update
                        sum += vl._weights[dPartial + az * dxyz] * (visibleActivation - visibleActivationPrev);
                    }
                }
        }

<<<<<<< HEAD
        // Determine highest cell activation and index
        if (sum > maxValue) {
            maxValue = sum;

            maxIndex = hc;
        }
    }

    // Output state
    _hiddenCs[address2(pos, _hiddenSize.x)] = maxIndex;
=======
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
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
}

void ImageEncoder::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

<<<<<<< HEAD
    // Create layers
=======
    // Initialize stimulus to 0
    cs.getQueue().enqueueFillBuffer(_hiddenActivations, static_cast<cl_float>(0.0f), 0, numHidden * sizeof(cl_float));

    // Forward
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

<<<<<<< HEAD
        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;

        // Projection constant
        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSize = numHidden * numWeightsPerHidden;

        // Create weight matrix for this visible layer and initialize randomly
        vl._weights = FloatBuffer(weightsSize);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < weightsSize; x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(ImageEncoder::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSize, cs._rng, cs._batchSize1);
#endif

        // Reconstruction buffer
        vl._visibleActivationsPrev = FloatBuffer(numVisible);

#ifdef KERNEL_DEBUG
        for (int x = 0; x < numVisible; x++)
            fillFloat(x, cs._rng, &vl._visibleActivationsPrev, 0.0f);
#else
        runKernel1(cs, std::bind(fillFloat, std::placeholders::_1, std::placeholders::_2, &vl._visibleActivationsPrev, 0.0f), numVisible, cs._rng, cs._batchSize1);
#endif
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif
}

void ImageEncoder::activate(ComputeSystem &cs, const std::vector<const FloatBuffer*> &inputActivations) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, inputActivations);
#else
    runKernel2(cs, std::bind(ImageEncoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputActivations), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
=======
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
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

    // Copy visible activations to prev
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

<<<<<<< HEAD
        int numVisibleColumns = vld._size.x * vld._size.y;
        int numVisible = numVisibleColumns * vld._size.z;
=======
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
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

#ifdef KERNEL_DEBUG
        for (int x = 0; x < numVisible; x++)
            copyFloat(x, cs._rng, inputActivations[vli], &vl._visibleActivationsPrev);
#else
        runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, inputActivations[vli], &vl._visibleActivationsPrev), numVisible, cs._rng, cs._batchSize1);
#endif
    }
}

void ImageEncoder::stepEnd(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleAs) {
    // Copy
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueCopyBuffer(visibleAs[vli], vl._visibleAs, 0, 0, vld._size.x * vld._size.y * vld._size.z * sizeof(cl_float));
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