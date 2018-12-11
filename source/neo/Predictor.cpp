// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

<<<<<<< HEAD
// Kernels
void Predictor::init(int pos, std::mt19937 &rng, int vli) {
    // Randomly initialize weights in range
	std::uniform_real_distribution<float> weightDist(-0.0001f, 0.0f);
=======
void Predictor::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
{
    _visibleLayerDescs = visibleLayerDescs;
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void Predictor::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs) {
    // Cache address calculations (taken from addressN functions)
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    // ------------------------------ Action ------------------------------

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        // Partially computed address of weight
        int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

<<<<<<< HEAD
        float sum = 0.0f;
        float count = 0.0f;
    
        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];
=======
        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

            // Center of projected position
            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

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
                    Int2 visiblePosition(x, y);

                    int visibleC = (*inputCs[vli])[address2(visiblePosition, vld._size.x)];

<<<<<<< HEAD
                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;
=======
            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, Vec2<cl_uint>(static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng))));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

                    sum += vl._weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }

            count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
        }

        // Normalize and save value for later
        float activation = sum / std::max(1.0f, count);

        if (activation > maxActivation) {
            maxActivation = activation;
            maxIndex = hc;
        }
    }

    _hiddenCs[address2(pos, _hiddenSize.x)] = maxIndex;
}

void Predictor::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCsPrev, const IntBuffer* hiddenTargetCs) {
    // Cache address calculations
    int dxy = _hiddenSize.x * _hiddenSize.y;
    int dxyz = dxy * _hiddenSize.z;

    float maxQ = -999999.0f;

    int hiddenIndex = address2(pos, _hiddenSize.x);

    // For each hidden unit
    for (int hc = 0; hc < _hiddenSize.z; hc++) {
        Int3 hiddenPosition(pos.x, pos.y, hc);

        // Partially computed address of weight
        int dPartial = hiddenPosition.x + hiddenPosition.y * _hiddenSize.x + hiddenPosition.z * dxy;

        float sum = 0.0f;
        float count = 0.0f;
    
        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            // Center of projected position
            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

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
                    Int2 visiblePosition(x, y);

                    int visibleCPrev = (*inputCsPrev[vli])[address2(visiblePosition, vld._size.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev * diam2;

                    sum += vl._weights[dPartial + az * dxyz]; // Used cached parts to compute weight address, equivalent to calling address4
                }

            // Count can be computed outside of loop, this is the value equavilent to count += 1.0f after each value increment
            count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
        }

        float delta = _alpha * ((hc == (*hiddenTargetCs)[hiddenIndex] ? 1.0f : 0.0f) - sigmoid(sum / std::max(1.0f, count)));

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            // Center of projected position
            Int2 visiblePositionCenter = project(pos, vl._hiddenToVisible);

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
                    Int2 visiblePosition(x, y);

                    int visibleCPrev = (*inputCsPrev[vli])[address2(visiblePosition, vld._size.x)];

                    // Final component of address
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleCPrev * diam2;

                    vl._weights[dPartial + az * dxyz] += delta;
                }
        }
    }
}

void Predictor::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
{
    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _visibleLayers.resize(_visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int numVisibleColumns = vld._size.x * vld._size.y;

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
        runKernel1(cs, std::bind(Predictor::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSize, cs._rng, cs._batchSize1);
#endif
    }
<<<<<<< HEAD

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif
}

void Predictor::activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

    // Forward kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            forward(Int2(x, y), cs._rng, visibleCs);
#else
    runKernel2(cs, std::bind(Predictor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
}

void Predictor::learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCsPrev, const IntBuffer* hiddenTargetCs) {
    // Learn kernel
#ifdef KERNEL_DEBUG
    for (int x = 0; x < _hiddenSize.x; x++)
        for (int y = 0; y < _hiddenSize.y; y++)
            learn(Int2(x, y), cs._rng, visibleCsPrev, hiddenTargetCs);
#else
    runKernel2(cs, std::bind(Predictor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCsPrev, hiddenTargetCs), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif
=======
}

void Predictor::writeToStream(ComputeSystem &cs, std::ostream &os) {
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

        std::vector<cl_int> visibleCs(numVisibleColumns);
        cs.getQueue().enqueueReadBuffer(vl._visibleCs, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
        os.write(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
    }
}

void Predictor::readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is) {
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

        std::vector<cl_int> visibleCs(numVisibleColumns);
        is.read(reinterpret_cast<char*>(visibleCs.data()), numVisibleColumns * sizeof(cl_int));
        vl._visibleCs = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numVisibleColumns * sizeof(cl_int));
        cs.getQueue().enqueueWriteBuffer(vl._visibleCs, CL_TRUE, 0, numVisibleColumns * sizeof(cl_int), visibleCs.data());
    }

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "pForward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "pInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "pLearn");
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
}