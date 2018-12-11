// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

<<<<<<< HEAD
// Kernels
void SparseCoder::init(int pos, std::mt19937 &rng, int vli) {
    // Initialize weights into uniform range
	std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);

    _visibleLayers[vli]._weights[pos] = weightDist(rng);
}

void SparseCoder::forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, bool firstIter) {
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

        // Accumulator
        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

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

                    int visibleIndex = address2(visiblePosition, vld._size.x);

                    int visibleC = (*inputCs[vli])[visibleIndex];

                    // Complete the partial address with final value needed
                    int az = visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visibleC * diam2;

                    // Rule is: sum += max(0, weight - prevActivation), found empirically to be better than truncated weight * (1.0 - prevActivation) update
                    sum += std::max(0.0f, vl._weights[dPartial + az * dxyz] - (firstIter ? 0.0f : vl._visibleActivations[visibleIndex]));
                }
        }

        int hiddenIndex = address3(hiddenPosition, Int2(_hiddenSize.x, _hiddenSize.y));

        if (firstIter) // Clear to new sum value if is first step
            _hiddenActivations[hiddenIndex] = sum;
        else
            _hiddenActivations[hiddenIndex] += sum; // Add on to sum (accumulate over sparse coding iterations)

        // Determine highest cell activation and index
        if (_hiddenActivations[hiddenIndex] > maxValue) {
            maxValue = _hiddenActivations[hiddenIndex];

            maxIndex = hc;
        }
    }

    // Output state
    _hiddenCs[address2(pos, _hiddenSize.x)] = maxIndex;
}

void SparseCoder::backward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleIndex = address2(pos, vld._size.x);

    Int3 visiblePosition(pos.x, pos.y, (*inputCs[vli])[visibleIndex]);

    // Project to hidden
    Int2 hiddenPositionCenter = project(pos, vl._visibleToHidden);

    // Additional addressing dimensions
    int diam = vld._radius * 2 + 1;
    int diam2 = diam * diam;

    // Accumulators
    float sum = 0.0f;
    float count = 0.0f;

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(std::max(0, hiddenPositionCenter.x - vl._reverseRadii.x), std::max(0, hiddenPositionCenter.y - vl._reverseRadii.y));
    Int2 iterUpperBound(std::min(_hiddenSize.x - 1, hiddenPositionCenter.x + vl._reverseRadii.x), std::min(_hiddenSize.y - 1, hiddenPositionCenter.y + vl._reverseRadii.y));

    for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
        for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
            Int2 hiddenPosition(x, y);

            // Next layer node's receptive field
            Int2 visibleFieldCenter = project(hiddenPosition, vl._hiddenToVisible);

            // Bounds of receptive field, clamped to input size
            Int2 fieldLowerBound(visibleFieldCenter.x - vld._radius, visibleFieldCenter.y - vld._radius);
            Int2 fieldUpperBound(visibleFieldCenter.y + vld._radius + 1, visibleFieldCenter.y + vld._radius + 1);

            // Check for containment
            if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                // Address cannot be easily partially computed here, compute fully (address4)
                int hiddenC = _hiddenCs[address2(hiddenPosition, _hiddenSize.x)];

                Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visiblePosition.z * diam2);

                sum += vl._weights[address4(wPos, _hiddenSize)];
                count += 1.0f;
            }
        }

    // Set normalized reconstruction value
    vl._visibleActivations[visibleIndex] = sum / std::max(1.0f, count);
}

void SparseCoder::learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int visibleIndex = address2(pos, vld._size.x);

    int inputC = (*inputCs[vli])[visibleIndex];

    // Project to hidden
    Int2 hiddenPositionCenter = project(pos, vl._visibleToHidden);

    int diam = vld._radius * 2 + 1;
    int diam2 = diam * diam;

    for (int vc = 0; vc < vld._size.z; vc++) {
        Int3 visiblePosition(pos.x, pos.y, vc);

        float sum = 0.0f;
        float count = 0.0f;

        Int2 iterLowerBound(std::max(0, hiddenPositionCenter.x - vl._reverseRadii.x), std::max(0, hiddenPositionCenter.y - vl._reverseRadii.y));
        Int2 iterUpperBound(std::min(_hiddenSize.x - 1, hiddenPositionCenter.x + vl._reverseRadii.x), std::min(_hiddenSize.y - 1, hiddenPositionCenter.y + vl._reverseRadii.y));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 hiddenPosition(x, y);
                
                // Next layer node's receptive field
                Int2 visibleFieldCenter = project(hiddenPosition, vl._hiddenToVisible);

                // Bounds of receptive field, clamped to input size
                Int2 fieldLowerBound(visibleFieldCenter.x - vld._radius, visibleFieldCenter.y - vld._radius);
                Int2 fieldUpperBound(visibleFieldCenter.x + vld._radius + 1, visibleFieldCenter.y + vld._radius + 1);

                // Check for containment
                if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                    // Address cannot be easily partially computed here, compute fully (address4)
                    int hiddenC = _hiddenCs[address2(hiddenPosition, _hiddenSize.x)];

                    Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visiblePosition.z * diam2);

                    sum += vl._weights[address4(wPos, _hiddenSize)];
                    count += 1.0f;
                }
            }

        // Weight increment
        float target = (vc == inputC ? 1.0f : 0.0f);

        float delta = _alpha * (target - sum / std::max(1.0f, count));

        for (int x = iterLowerBound.x; x <= iterUpperBound.x; x++)
            for (int y = iterLowerBound.y; y <= iterUpperBound.y; y++) {
                Int2 hiddenPosition(x, y);

                // Next layer node's receptive field
                Int2 visibleFieldCenter = project(hiddenPosition, vl._hiddenToVisible);

                // Bounds of receptive field, clamped to input size
                Int2 fieldLowerBound(visibleFieldCenter.x - vld._radius, visibleFieldCenter.y - vld._radius);
                Int2 fieldUpperBound(visibleFieldCenter.x + vld._radius + 1, visibleFieldCenter.y + vld._radius + 1);

                // Check for containment
                if (inBounds(pos, fieldLowerBound, fieldUpperBound)) {
                    // Address cannot be easily partially computed here, compute fully (address4)
                    int hiddenC = _hiddenCs[address2(hiddenPosition, _hiddenSize.x)];

                    Int4 wPos(hiddenPosition.x, hiddenPosition.y, hiddenC, visiblePosition.x - fieldLowerBound.x + (visiblePosition.y - fieldLowerBound.y) * diam + visiblePosition.z * diam2);

                    vl._weights[address4(wPos, _hiddenSize)] += delta;
                }
            }
    }
}

void SparseCoder::createRandom(ComputeSystem &cs,
    const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs)
=======
void SparseCoder::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    std::mt19937 &rng)
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
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
        int numVisible = numVisibleColumns * vld._size.z;

<<<<<<< HEAD
        // Projection constants
        vl._visibleToHidden = Float2(static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y));
=======
        vl._visibleToHidden = Float2(static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y));

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        vl._reverseRadii = Int2(static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

        vl._hiddenToVisible = Float2(static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y));

        vl._reverseRadii = Int2(static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1));

        int diam = vld._radius * 2 + 1;

        int numWeightsPerHidden = diam * diam * vld._size.z;

        int weightsSize = numHidden * numWeightsPerHidden;

        // Create weight matrix for this visible layer and initialize randomly
        vl._weights = FloatBuffer(weightsSize);

<<<<<<< HEAD
#ifdef KERNEL_DEBUG
        for (int x = 0; x < weightsSize; x++)
            init(x, cs._rng, vli);
#else
        runKernel1(cs, std::bind(SparseCoder::initKernel, std::placeholders::_1, std::placeholders::_2, this, vli), weightsSize, cs._rng, cs._batchSize1);
#endif
=======
            initWeightsKernel.setArg(argIndex++, vl._weights);
            initWeightsKernel.setArg(argIndex++, Vec2<cl_uint>(static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng))));
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

        // Reconstruction buffer
        vl._visibleActivations = FloatBuffer(numVisibleColumns);
    }

    // Hidden Cs
    _hiddenCs = IntBuffer(numHiddenColumns);
    _hiddenCsPrev = IntBuffer(numHiddenColumns);

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCs, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        fillInt(x, cs._rng, &_hiddenCsPrev, 0);
#else
    runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCsPrev, 0), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Hidden activations
<<<<<<< HEAD
    _hiddenActivations = FloatBuffer(numHidden);
=======
    _hiddenActivations = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, numHidden * sizeof(cl_float));

    // Create kernels
    _forwardKernel = cl::Kernel(prog.getProgram(), "scForward");
    _backwardPartialKernel = cl::Kernel(prog.getProgram(), "scBackwardPartial");
    _backwardKernel = cl::Kernel(prog.getProgram(), "scBackward");
    _inhibitKernel = cl::Kernel(prog.getProgram(), "scInhibit");
    _learnKernel = cl::Kernel(prog.getProgram(), "scLearn");
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
}

void SparseCoder::activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    int numHiddenColumns = _hiddenSize.x * _hiddenSize.y;
    int numHidden = numHiddenColumns * _hiddenSize.z;

#ifdef KERNEL_DEBUG
    for (int x = 0; x < numHiddenColumns; x++)
        copyInt(x, cs._rng, &_hiddenCs, &_hiddenCsPrev);
#else
    runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_hiddenCs, &_hiddenCsPrev), numHiddenColumns, cs._rng, cs._batchSize1);
#endif

    // Sparse coding iterations: forward, reconstruct, repeat
    for (int it = 0; it < _explainIters; it++) {
        bool firstIter = it == 0;

#ifdef KERNEL_DEBUG
        for (int x = 0; x < _hiddenSize.x; x++)
            for (int y = 0; y < _hiddenSize.y; y++)
                forward(Int2(x, y), cs._rng, visibleCs, firstIter);
#else
        runKernel2(cs, std::bind(SparseCoder::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, firstIter), Int2(_hiddenSize.x, _hiddenSize.y), cs._rng, cs._batchSize2);
#endif

<<<<<<< HEAD
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

#ifdef KERNEL_DEBUG
            for (int x = 0; x < vld._size.x; x++)
                for (int y = 0; y < vld._size.y; y++)
                    backward(Int2(x, y), cs._rng, visibleCs, vli);
#else
            runKernel2(cs, std::bind(SparseCoder::backwardKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
=======
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
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
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

<<<<<<< HEAD
void SparseCoder::learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs) {
    // Final reconstruction + learning
=======
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
    
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

<<<<<<< HEAD
#ifdef KERNEL_DEBUG
        for (int x = 0; x < vld._size.x; x++)
            for (int y = 0; y < vld._size.y; y++)
                learn(Int2(x, y), cs._rng, visibleCs, vli);
#else
        runKernel2(cs, std::bind(SparseCoder::learnKernel, std::placeholders::_1, std::placeholders::_2, this, visibleCs, vli), Int2(vld._size.x, vld._size.y), cs._rng, cs._batchSize2);
#endif
=======
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
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
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