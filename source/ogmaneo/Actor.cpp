// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace ogmaneo;

void Actor::forward(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value ---

    float value = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyOHVs(*inputCs[vli], hiddenColumnIndex, vld.size.z);
        count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
    }

    hiddenValues[hiddenColumnIndex] = value / count;

    // --- Action ---

    std::vector<float> activations(hiddenSize.z);
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
        }

        sum /= count;

        activations[hc] = sum;

        maxActivation = std::max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        activations[hc] = std::exp(activations[hc] - maxActivation);
        
        total += activations[hc];
    }

    std::uniform_real_distribution<float> cuspDist(0.0f, total);

    float cusp = cuspDist(rng);

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        sumSoFar += activations[hc];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }
    
    hiddenCs[hiddenColumnIndex] = selectIndex;
}

void Actor::learn(
    const Int2 &pos,
    std::mt19937 &rng,
    const std::vector<const IntBuffer*> &inputCsPrev,
    const IntBuffer* hiddenTargetCsPrev,
    const FloatBuffer* hiddenValuesPrev,
    float q,
    float g,
    bool mimic
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value Prev ---

    float newValue = q + g * hiddenValues[hiddenColumnIndex];

    float value = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyOHVs(*inputCsPrev[vli], hiddenColumnIndex, vld.size.z);
        count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
    }

    value /= count;

    float tdErrorValue = newValue - value;
    
    float deltaValue = alpha * tdErrorValue;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.valueWeights.deltaOHVs(*inputCsPrev[vli], deltaValue, hiddenColumnIndex, vld.size.z);
    }

    // --- Action ---

    float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

    int targetC = (*hiddenTargetCsPrev)[hiddenColumnIndex];

    std::vector<float> activations(hiddenSize.z);
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyOHVs(*inputCsPrev[vli], hiddenIndex, vld.size.z);
        }

        sum /= count;

        activations[hc] = sum;

        maxActivation = std::max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        activations[hc] = std::exp(activations[hc] - maxActivation);
        
        total += activations[hc];
    }
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float deltaAction = (mimic ? beta : (tdErrorAction > 0.0f ? beta : -beta)) * ((hc == targetC ? 1.0f : 0.0f) - activations[hc] / std::max(0.0001f, total));

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.actionWeights.deltaOHVs(*inputCsPrev[vli], deltaAction, hiddenIndex, vld.size.z);
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

    std::uniform_real_distribution<float> weightDist(-0.01f, 0.01f);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, Int3(hiddenSize.x, hiddenSize.y, 1), vld.radius, vl.valueWeights);
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.actionWeights);

        for (int i = 0; i < vl.valueWeights.nonZeroValues.size(); i++)
            vl.valueWeights.nonZeroValues[i] = 0.0f;

        for (int i = 0; i < vl.actionWeights.nonZeroValues.size(); i++)
            vl.actionWeights.nonZeroValues[i] = weightDist(cs.rng);
    }

    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenValues = FloatBuffer(numHiddenColumns, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i].inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i].inputCs[vli] = IntBuffer(numVisibleColumns);
        }

        historySamples[i].hiddenTargetCsPrev = IntBuffer(numHiddenColumns);

        historySamples[i].hiddenValuesPrev = FloatBuffer(numHiddenColumns);
    }
}

void Actor::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenTargetCsPrev,
    float reward,
    bool learnEnabled,
    bool mimic
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    runKernel2(cs, std::bind(Actor::forwardKernel, std::placeholders::_1, std::placeholders::_2, this, inputCs), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples.front();

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            // Copy visible Cs
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[vli], &s.inputCs[vli]), numVisibleColumns, cs.rng, cs.batchSize1);
        }

        // Copy hidden Cs
        runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, hiddenTargetCsPrev, &s.hiddenTargetCsPrev), numHiddenColumns, cs.rng, cs.batchSize1);

        // Copy hidden values
        runKernel1(cs, std::bind(copyFloat, std::placeholders::_1, std::placeholders::_2, &hiddenValues, &s.hiddenValuesPrev), numHiddenColumns, cs.rng, cs.batchSize1);

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > minSteps + 1) {
        std::uniform_int_distribution<int> historyDist(minSteps, historySize - 1);

        for (int it = 0; it < historyIters; it++) {
            int historyIndex = historyDist(cs.rng);

            const HistorySample &sPrev = historySamples[historyIndex + 1];
            const HistorySample &s = historySamples[historyIndex];

            // Compute (partial) values, rest is completed in the kernel
            float q = 0.0f;
            float g = 1.0f;

            for (int t = historyIndex; t >= 0; t--) {
                q += historySamples[t].reward * g;

                g *= gamma;
            }

            // Learn kernel
            runKernel2(cs, std::bind(Actor::learnKernel, std::placeholders::_1, std::placeholders::_2, this, constGet(sPrev.inputCs), &s.hiddenTargetCsPrev, &sPrev.hiddenValuesPrev, q, g, mimic), Int2(hiddenSize.x, hiddenSize.y), cs.rng, cs.batchSize2);
        }
    }
}

void Actor::writeToStream(
    std::ostream &os
) const {
    os.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(Int3));

    os.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    os.write(reinterpret_cast<const char*>(&beta), sizeof(float));
    os.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    os.write(reinterpret_cast<const char*>(&minSteps), sizeof(int));
    os.write(reinterpret_cast<const char*>(&historyIters), sizeof(int));

    writeBufferToStream(os, &hiddenCs);

    writeBufferToStream(os, &hiddenValues);

    int numVisibleLayers = visibleLayers.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        os.write(reinterpret_cast<const char*>(&vld), sizeof(VisibleLayerDesc));

        writeSMToStream(os, vl.valueWeights);
        writeSMToStream(os, vl.actionWeights);
    }

    os.write(reinterpret_cast<const char*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    os.write(reinterpret_cast<const char*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    os.write(reinterpret_cast<const char*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writeBufferToStream(os, &s.inputCs[vli]);

        writeBufferToStream(os, &s.hiddenTargetCsPrev);

        writeBufferToStream(os, &s.hiddenValuesPrev);

        os.write(reinterpret_cast<const char*>(&s.reward), sizeof(float));
    }
}

void Actor::readFromStream(
    std::istream &is
) {
    is.read(reinterpret_cast<char*>(&hiddenSize), sizeof(Int3));

    is.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&minSteps), sizeof(int));
    is.read(reinterpret_cast<char*>(&historyIters), sizeof(int));

    readBufferFromStream(is, &hiddenCs);

    readBufferFromStream(is, &hiddenValues);

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

        readSMFromStream(is, vl.valueWeights);
        readSMFromStream(is, vl.actionWeights);
    }

    is.read(reinterpret_cast<char*>(&historySize), sizeof(int));

    int numHistorySamples;

    is.read(reinterpret_cast<char*>(&numHistorySamples), sizeof(int));

    int historyStart;

    is.read(reinterpret_cast<char*>(&historyStart), sizeof(int));

    historySamples.resize(numHistorySamples);
    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        s.inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            readBufferFromStream(is, &s.inputCs[vli]);

        readBufferFromStream(is, &s.hiddenTargetCsPrev);

        readBufferFromStream(is, &s.hiddenValuesPrev);

        is.read(reinterpret_cast<char*>(&s.reward), sizeof(float));
    }
}