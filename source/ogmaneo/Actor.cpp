// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

// Kernels
void Actor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= std::max(1, count);

        if (sum > maxActivation) {
            maxActivation = sum;

            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    hiddenActivationsPrev[hiddenColumnIndex] = hiddenActivations[hiddenColumnIndex];
    hiddenActivations[hiddenColumnIndex] = maxActivation;
}

void Actor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const IntBuffer* hiddenCsPrev,
    float reward
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndexPrev = address3(Int3(pos.x, pos.y, (*hiddenCsPrev)[hiddenColumnIndex]), hiddenSize);

    float newValue = reward + gamma * hiddenActivations[hiddenColumnIndex];

    float tdError = newValue - hiddenActivationsPrev[hiddenColumnIndex];

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.traces.setTraceOHVs(vl.inputCsPrev, hiddenIndexPrev, vld.size.z);
    }

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.weights.deltaTraceOHVs(vl.traces, alpha * tdError, hiddenIndex, vld.size.z, traceDecay);
        }
    }
}

void Actor::initRandom(
    ComputeSystem &cs,
    const Int3 &hiddenSize,
    int historyCapacity,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        vl.traces = vl.weights;

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++) {
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);
            vl.traces.nonZeroValues[i] = 0.0f;
        }

        vl.inputCsPrev = IntBuffer(numVisibleColumns, 0);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenColumns, 0.0f);
    hiddenActivationsPrev = FloatBuffer(numHiddenColumns, 0.0f);
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenCsPrev,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Forward kernel
#ifdef KERNELNOTHREAD
    for (int x = 0; x < hiddenSize.x; x++)
        for (int y = 0; y < hiddenSize.y; y++)
            forward(Int2(x, y), cs.rng, inputCs);
#else
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
#endif

    // Learn (if have sufficient samples)
    if (learnEnabled) {
        // Learn kernel
#ifdef KERNELNOTHREAD
        for (int x = 0; x < hiddenSize.x; x++)
            for (int y = 0; y < hiddenSize.y; y++)
                learn(Int2(x, y), cs.rng, hiddenCsPrev, reward);
#else
        runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, hiddenCsPrev, reward), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
#endif
    }

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

#ifdef KERNEL_NO_THREAD
        for (int x = 0; x < inputCs[vli]->size(); x++)
            copyInt(x, cs.rng, inputCs[vli], &vl.inputCsPrev);
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &vl.inputCsPrev), inputCs[vli]->size(), cs.rng, cs.batchSize1);
#endif
    }
}

void Actor::writeToStream(
    std::ostream &os
) const {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));

    writeBufferToStream(os, &hiddenCs);

    writeBufferToStream(os, &hiddenActivations);
    writeBufferToStream(os, &hiddenActivationsPrev);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);

        writeBufferToStream(os, &vl.inputCsPrev);
    }
}

void Actor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));

    readBufferFromStream(is, &hiddenCs);

    readBufferFromStream(is, &hiddenActivations);
    readBufferFromStream(is, &hiddenActivationsPrev);

    int numVisibleLayers;
    
    is.read(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        is.read(reinterpret_cast<char*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        readSMFromStream(is, vl.weights);

        readBufferFromStream(is, &vl.inputCsPrev);
    }
}