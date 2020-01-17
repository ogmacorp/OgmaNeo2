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

    hiddenActivations[hiddenColumnIndex] = maxActivation;
}

void Actor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCsPrev,
    const IntBuffer* hiddenCsPrev,
    float q,
    float g
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, (*hiddenCsPrev)[hiddenColumnIndex]), hiddenSize);

    float newValue = q + g * hiddenActivations[hiddenColumnIndex];

    float sum = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        sum += vl.weights.multiplyOHVs(*inputCsPrev[vli], hiddenIndex, vld.size.z);
        count += vl.weights.count(hiddenIndex) / vld.size.z;
    }

    sum /= std::max(1, count);

    float delta = alpha * (newValue - sum);

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.weights.deltaOHVs(*inputCsPrev[vli], delta, hiddenIndex, vld.size.z);
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

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = weightDist(cs.rng);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenColumns, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i] = std::make_shared<HistorySample>();

        historySamples[i]->inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i]->inputCs[vli] = IntBuffer(numVisibleColumns);
        }

        historySamples[i]->hiddenCsPrev = IntBuffer(numHiddenColumns);
    }
}

const Actor &Actor::operator=(
    const Actor &other
) {
    hiddenSize = other.hiddenSize;

    hiddenCs = other.hiddenCs;

    hiddenActivations = other.hiddenActivations;

    visibleLayerDescs = other.visibleLayerDescs;
    visibleLayers = other.visibleLayers;

    alpha = other.alpha;
    gamma = other.gamma;

    historySize = other.historySize;

    historySamples.resize(other.historySamples.size());

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        (*historySamples[t]) = (*other.historySamples[t]);
    }

    return *this;
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

    // Add sample
    if (historySize == historySamples.size()) {
        // Circular buffer swap
        std::shared_ptr<HistorySample> temp = historySamples.front();

        for (int i = 0; i < historySamples.size() - 1; i++)
            historySamples[i] = historySamples[i + 1];

        historySamples.back() = temp;
    }

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = *historySamples[historySize - 1];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            // Copy visible Cs
#ifdef KERNELNOTHREAD
            for (int x = 0; x < numVisibleColumns; x++)
                copyInt(x, cs.rng, inputCs[vli], &s.inputCs[vli]);
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &s.inputCs[vli]), numVisibleColumns, cs.rng, cs.batchSize1);
#endif
        }

        // Copy hidden Cs
#ifdef KERNELNOTHREAD
        for (int x = 0; x < numHiddenColumns; x++)
            copyInt(x, cs.rng, hiddenCsPrev, &s.hiddenCsPrev);
#else
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, hiddenCsPrev, &s.hiddenCsPrev), numHiddenColumns, cs.rng, cs.batchSize1);
#endif

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > 1) {
        const HistorySample &sPrev = *historySamples[0];
        const HistorySample &s = *historySamples[1];

        // Compute (partial) values, rest is completed in the kernel
        float q = 0.0f;
        float g = 1.0f;

        for (int t = 1; t < historySize; t++) {
            q += historySamples[t]->reward * g;

            g *= gamma;
        }

        // Learn kernel
#ifdef KERNELNOTHREAD
        for (int x = 0; x < hiddenSize.x; x++)
            for (int y = 0; y < hiddenSize.y; y++)
                learn(Int2(x, y), cs.rng, constGet(sPrev.inputCs), &s.hiddenCsPrev, q, g);
#else
        runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, constGet(sPrev.inputCs), &s.hiddenCsPrev, q, g), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
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

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.weights);
    }

    os.write(reinterpret_cast<const char*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = *historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writeBufferToStream(os, &s.inputCs[vli]);

        writeBufferToStream(os, &s.hiddenCsPrev);

        os.write(reinterpret_cast<const char*>(&s.reward), sizeof(float));
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
    }

    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    historySamples.resize(numHistorySamples);

    for (int t = 0; t < historySamples.size(); t++) {
        historySamples[t] = std::make_shared<HistorySample>();

        HistorySample &s = *historySamples[t];

        s.inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            readBufferFromStream(is, &s.inputCs[vli]);

        readBufferFromStream(is, &s.hiddenCsPrev);

        is.read(reinterpret_cast<char*>(&s.reward), sizeof(float));
    }
}