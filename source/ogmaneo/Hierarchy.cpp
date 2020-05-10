// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace ogmaneo;

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    scLayers.resize(layerDescs.size());
    pLayers.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    historySizes.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l].temporalHorizon : layerDescs[l].temporalHorizon);

        historySizes[l].resize(histories[l].size());
		
        // Create sparse coder visible layer descriptors
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    scVisibleLayerDescs[index].size = inputSizes[i];
                    scVisibleLayerDescs[index].radius = layerDescs[l].ffRadius;
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < histories[l].size(); v++) {
				int i = v / layerDescs[l].temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
			}

            // Predictors
            pLayers[l].resize(inputSizes.size());
            aLayers.resize(inputSizes.size());

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Actor visible layer descriptors
            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs(1);

            aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            aVisibleLayerDescs[0].radius = layerDescs[l].aRadius;

            if (l < scLayers.size() - 1)
                aVisibleLayerDescs.push_back(aVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::prediction) {
                    pLayers[l][p] = std::make_unique<Predictor>();

                    pLayers[l][p]->initRandom(cs, inputSizes[p], pVisibleLayerDescs);
                }
                else if (inputTypes[p] == InputType::action) {
                    aLayers[p] = std::make_unique<Actor>();

                    aLayers[p]->initRandom(cs, inputSizes[p], layerDescs[l].historyCapacity, aVisibleLayerDescs);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                scVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                scVisibleLayerDescs[t].radius = layerDescs[l].ffRadius;
            }

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

			for (int v = 0; v < histories[l].size(); v++) {
                histories[l][v] = std::make_shared<IntBuffer>(inSize, 0);

                historySizes[l][v] = inSize;
            }

            pLayers[l].resize(layerDescs[l].ticksPerUpdate);

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p] = std::make_unique<Predictor>();

                pLayers[l][p]->initRandom(cs, layerDescs[l - 1].hiddenSize, pVisibleLayerDescs);
            }
        }
		
        // Create the sparse coding layer
        scLayers[l].initRandom(cs, layerDescs[l].hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    scLayers = other.scLayers;

    historySizes = other.historySizes;
    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;

    pLayers.resize(other.pLayers.size());
    histories.resize(other.histories.size());

    for (int l = 0; l < scLayers.size(); l++) {
        pLayers[l].resize(other.pLayers[l].size());

        for (int v = 0; v < pLayers[l].size(); v++) {
            if (other.pLayers[l][v] != nullptr) {
                pLayers[l][v] = std::make_unique<Predictor>();

                (*pLayers[l][v]) = (*other.pLayers[l][v]);
            }
            else
                pLayers[l][v] = nullptr;
        }

        histories[l].resize(other.histories[l].size());

        for (int v = 0; v < histories[l].size(); v++) {
            histories[l][v] = std::make_unique<IntBuffer>();
            
            (*histories[l][v]) = (*other.histories[l][v]);
        }
    }

    aLayers.resize(inputSizes.size());
    
    for (int v = 0; v < aLayers.size(); v++) {
        if (other.aLayers[v] != nullptr) {
            aLayers[v] = std::make_unique<Actor>();

            (*aLayers[v]) = (*other.aLayers[v]);
        }
        else
            aLayers[v] = nullptr;
    }

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    assert(inputCs.size() == inputSizes.size());

    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = histories.front().size() / inputSizes.size();

        std::vector<std::shared_ptr<IntBuffer>> lasts(inputSizes.size());
        
        for (int i = 0; i < inputSizes.size(); i++)
            lasts[i] = histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < inputSizes.size(); i++) {
                // Shift
                histories.front()[t + temporalHorizon * i] = histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < inputSizes.size(); i++) {
            assert(inputSizes[i].x * inputSizes[i].y == inputCs[i]->size());
            
            // Copy
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[i], lasts[i].get()), inputCs[i]->size(), cs.rng, cs.batchSize1);

            histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    updates.clear();
    updates.resize(scLayers.size(), false);

    // Forward
    for (int l = 0; l < scLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            // Activate sparse coder
            scLayers[l].step(cs, constGet(histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = histories[lNext].size();

                std::shared_ptr<IntBuffer> last = histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    histories[lNext][t] = histories[lNext][t - 1];

                // Copy
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &scLayers[l].getHiddenCs(), last.get()), scLayers[l].getHiddenCs().size(), cs.rng, cs.batchSize1);

                histories[lNext].front() = last;

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = scLayers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            std::vector<const IntBuffer*> feedBackCs(l < scLayers.size() - 1 ? 2 : 1);

            feedBackCs[0] = &scLayers[l].getHiddenCs();

            if (l < scLayers.size() - 1) {
                assert(pLayers[l + 1][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]] != nullptr);

                feedBackCs[1] = &pLayers[l + 1][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]]->getHiddenCs();
            }

            // Step actor layers
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (pLayers[l][p] != nullptr) {
                    if (learnEnabled)
                        pLayers[l][p]->learn(cs, l == 0 ? inputCs[p] : histories[l][p].get());

                    pLayers[l][p]->activate(cs, feedBackCs);
                }
            }

            if (l == 0) {
                // Step actors
                for (int p = 0; p < aLayers.size(); p++) {
                    if (aLayers[p] != nullptr)
                        aLayers[p]->step(cs, feedBackCs, inputCs[p], reward, learnEnabled, mimic);
                }
            }
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(updates.data()), updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(ticks.data()), ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < historySizes[l].size(); i++)
            writeBufferToStream(os, histories[l][i].get());

        scLayers[l].writeToStream(os);

        // Predictors
        for (int v = 0; v < pLayers[l].size(); v++) {
            char exists = pLayers[l][v] != nullptr;

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists)
                pLayers[l][v]->writeToStream(os);
        }
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        char exists = aLayers[v] != nullptr;

        os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

        if (exists)
            aLayers[v]->writeToStream(os);
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;

    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(inputSizes.data()), numInputs * sizeof(Int3));

    scLayers.resize(numLayers);
    pLayers.resize(numLayers);

    ticks.resize(numLayers);

    histories.resize(numLayers);
    historySizes.resize(numLayers);
    
    ticksPerUpdate.resize(numLayers);

    updates.resize(numLayers);

    is.read(reinterpret_cast<char*>(updates.data()), updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(ticks.data()), ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(ticksPerUpdate.data()), ticksPerUpdate.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(historySizes[l].data()), numHistorySizes * sizeof(int));

        histories[l].resize(numHistorySizes);

        for (int i = 0; i < historySizes[l].size(); i++) {
            histories[l][i] = std::make_shared<IntBuffer>();

            readBufferFromStream(is, histories[l][i].get());
        }

        scLayers[l].readFromStream(is);
        
        pLayers[l].resize(l == 0 ? inputSizes.size() : ticksPerUpdate[l]);

        // Predictors
        for (int v = 0; v < pLayers[l].size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                pLayers[l][v] = std::make_unique<Predictor>();
                pLayers[l][v]->readFromStream(is);
            }
            else
                pLayers[l][v] = nullptr;
        }
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int v = 0; v < aLayers.size(); v++) {
        char exists;

        is.read(reinterpret_cast<char*>(&exists), sizeof(char));

        if (exists) {
            aLayers[v] = std::make_unique<Actor>();
            aLayers[v]->readFromStream(is);
        }
        else
            aLayers[v] = nullptr;
    }
}

void Hierarchy::getState(
    State &state
) const {
    int numLayers = scLayers.size();

    state.hiddenCs.resize(numLayers);
    state.hiddenCsPrev.resize(numLayers);
    state.histories.resize(numLayers);
    state.predHiddenCs.resize(numLayers);
    state.predInputCsPrev.resize(numLayers);
    state.predInputCsPrevPrev.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        state.hiddenCs[l] = scLayers[l].getHiddenCs();
        state.hiddenCsPrev[l] = scLayers[l].getHiddenCsPrev();

        state.histories[l].resize(historySizes[l].size());

        for (int i = 0; i < historySizes[l].size(); i++)
            state.histories[l][i] = *histories[l][i];

        state.predHiddenCs[l].resize(pLayers[l].size());
        state.predInputCsPrev[l].resize(pLayers[l].size());
        state.predInputCsPrevPrev[l].resize(pLayers[l].size());

        for (int j = 0; j < pLayers[l].size(); j++) {
            state.predHiddenCs[l][j] = pLayers[l][j]->getHiddenCs();

            state.predInputCsPrev[l][j].resize(pLayers[l][j]->getNumVisibleLayers());
            state.predInputCsPrevPrev[l][j].resize(pLayers[l][j]->getNumVisibleLayers());

            for (int v = 0; v < pLayers[l][j]->getNumVisibleLayers(); v++) {
                state.predInputCsPrev[l][j][v] = pLayers[l][j]->getVisibleLayer(v).inputCsPrev;
                state.predInputCsPrevPrev[l][j][v] = pLayers[l][j]->getVisibleLayer(v).inputCsPrevPrev;
            }
        }
    }

    state.ticks = ticks;
    state.updates = updates;
}

void Hierarchy::setState(
    const State &state
) {
    int numLayers = scLayers.size();

    for (int l = 0; l < numLayers; l++) {
        scLayers[l].hiddenCs = state.hiddenCs[l];
        scLayers[l].hiddenCsPrev = state.hiddenCsPrev[l];

        for (int i = 0; i < historySizes[l].size(); i++)
            *histories[l][i] = state.histories[l][i];

        for (int j = 0; j < pLayers[l].size(); j++) {
            pLayers[l][j]->hiddenCs = state.predHiddenCs[l][j];

            for (int v = 0; v < pLayers[l][j]->getNumVisibleLayers(); v++) {
                pLayers[l][j]->visibleLayers[v].inputCsPrev = state.predInputCsPrev[l][j][v];
                pLayers[l][j]->visibleLayers[v].inputCsPrevPrev = state.predInputCsPrevPrev[l][j][v];
            }
        }
    }

    ticks = state.ticks;
    updates = state.updates;
}