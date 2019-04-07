// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace ogmaneo;

void State::initZero(
    const Hierarchy &h
) {
    int numLayers = h.getNumLayers();

    _ticks.assign(numLayers, 0);
    _updates.resize(numLayers, false);

    _histories.resize(numLayers);
    _predictions.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        _histories[l].resize(h._historySizes[l].size());

        for (int v = 0; v < _histories[l].size(); v++)
            _histories[l][v] = IntBuffer(h._historySizes[l][v], 0);
            
        const std::vector<std::unique_ptr<Predictor>> &predictors = h.getPLayer(l);

        _predictions[l].resize(predictors.size());

        for (int v = 0; v < _predictions[l].size(); v++) {
            if (predictors[v] != nullptr)
                _predictions[l][v] = IntBuffer(predictors[v]->getHiddenSize().x * predictors[v]->getHiddenSize().y, 0);
            else
                _predictions[l][v].clear();
        }
    }

    _predictionsPrev = _predictions;
}

void State::writeToStream(
    std::ostream &os
) const {
    int numLayers = _histories.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _histories[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        for (int i = 0; i < numHistorySizes; i++)
            writeBufferToStream(os, &_histories[l][i]);

        int numPredictions = _predictions[l].size();

        os.write(reinterpret_cast<const char*>(&numPredictions), sizeof(int));

        for (int i = 0; i < numPredictions; i++) {
            writeBufferToStream(os, &_predictions[l][i]);
            writeBufferToStream(os, &_predictionsPrev[l][i]);
        }
    }
}

void State::readFromStream(
    std::istream &is
) {
    int numLayers;

    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    _updates.resize(numLayers);
    _ticks.resize(numLayers);
    _histories.resize(numLayers);
    _predictions.resize(numLayers);

    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(char));
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;

        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));

        _histories[l].resize(numHistorySizes);

        for (int i = 0; i < numHistorySizes; i++)
            readBufferFromStream(is, &_histories[l][i]);

        int numPredictions;

        is.read(reinterpret_cast<char*>(&numPredictions), sizeof(int));

        _predictions[l].resize(numPredictions);

        for (int i = 0; i < numPredictions; i++) {
            readBufferFromStream(is, &_predictions[l][i]);
            readBufferFromStream(is, &_predictionsPrev[l][i]);
        }
    }
}

void Hierarchy::initRandom(
    ComputeSystem &cs,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs
) {
    // Create layers
    _scLayers.resize(layerDescs.size());
    _pLayers.resize(layerDescs.size());

    
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    // Cache input sizes
    _inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        _historySizes[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);
		
        // Create sparse coder visible layer descriptors
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
                }
            }
            
            // Initialize history buffers
			for (int v = 0; v < _historySizes[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
                _historySizes[l][v] = inSize;
			}

            // Predictors
            _pLayers[l].resize(inputSizes.size());

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;

            if (l < _scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::_predict) {
                    _pLayers[l][p] = std::make_unique<Predictor>();

                    _pLayers[l][p]->initRandom(cs, inputSizes[p], pVisibleLayerDescs);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._scRadius;
            }

            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _historySizes[l].size(); v++)
                _historySizes[l][v] = inSize;

            _pLayers[l].resize(layerDescs[l]._ticksPerUpdate);

            // Predictor visible layer descriptors
            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(1);

            pVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;

            if (l < _scLayers.size() - 1)
                pVisibleLayerDescs.push_back(pVisibleLayerDescs[0]);

            // Create actors
            for (int p = 0; p < _pLayers[l].size(); p++) {
                _pLayers[l][p] = std::make_unique<Predictor>();

                _pLayers[l][p]->initRandom(cs, layerDescs[l - 1]._hiddenSize, pVisibleLayerDescs);
            }
        }
		
        // Create the sparse coding layer
        _scLayers[l].initRandom(cs, layerDescs[l]._hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    _scLayers = other._scLayers;

    _historySizes = other._historySizes;
    _ticksPerUpdate = other._ticksPerUpdate;
    _inputSizes = other._inputSizes;

    _pLayers.resize(other._pLayers.size());

    for (int l = 0; l < _scLayers.size(); l++) {
        _pLayers[l].resize(other._pLayers[l].size());

        for (int v = 0; v < _pLayers[l].size(); v++) {
            if (other._pLayers[l][v] != nullptr) {
                _pLayers[l][v] = std::make_unique<Predictor>();

                (*_pLayers[l][v]) = (*other._pLayers[l][v]);
            }
            else
                _pLayers[l][v] = nullptr;
        }
    }

    return *this;
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<const IntBuffer*> &inputCs,
    State &state,
    bool learnEnabled
) {
    assert(inputCs.size() == _inputSizes.size());

    std::vector<int> ticksPrev = state._ticks;

    state._predictionsPrev = state._predictions;

    // First tick is always 0
    state._ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = state._histories.front().size() / _inputSizes.size();

        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < _inputSizes.size(); i++) {
                // Shift
                state._histories.front()[t + temporalHorizon * i] = state._histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < _inputSizes.size(); i++) {
            assert(_inputSizes[i].x * _inputSizes[i].y == inputCs[i]->size());
            
            state._histories.front()[0 + temporalHorizon * i] = *inputCs[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    state._updates.clear();
    state._updates.resize(_scLayers.size(), false);

    std::vector<IntBuffer> hiddenCsPrev(_scLayers.size());

    // Forward
    for (int l = 0; l < _scLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || state._ticks[l] >= _ticksPerUpdate[l]) {
            // Reset tick
            state._ticks[l] = 0;

            // Updated
            state._updates[l] = true;

            hiddenCsPrev[l] = _scLayers[l].getHiddenCs();
            
            // Activate sparse coder
            _scLayers[l].step(cs, constGet(state._histories[l]), learnEnabled);

            // Add to next layer's history
            if (l < _scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = state._histories[lNext].size();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    state._histories[lNext][t] = state._histories[lNext][t - 1];

                state._histories[lNext].front() = _scLayers[l].getHiddenCs();

                state._ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _scLayers.size() - 1; l >= 0; l--) {
        if (state._updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            std::vector<const IntBuffer*> feedBackCs(l < _scLayers.size() - 1 ? 2 : 1);
            std::vector<const IntBuffer*> feedBackCsPrev(feedBackCs.size());

            feedBackCs[0] = &_scLayers[l].getHiddenCs();

            feedBackCsPrev[0] = &hiddenCsPrev[l];

            if (l < _scLayers.size() - 1) {
                assert(_pLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - state._ticks[l + 1]] != nullptr);

                feedBackCs[1] = &state._predictions[l + 1][_ticksPerUpdate[l + 1] - 1 - state._ticks[l + 1]];
                feedBackCsPrev[1] = &state._predictionsPrev[l + 1][_ticksPerUpdate[l + 1] - 1 - ticksPrev[l + 1]];
            }

            // Step actor layers
            for (int p = 0; p < _pLayers[l].size(); p++) {
                if (_pLayers[l][p] != nullptr) {
                    if (learnEnabled)
                        _pLayers[l][p]->learn(cs, l == 0 ? inputCs[p] : &state._histories[l][p], feedBackCsPrev);

                    _pLayers[l][p]->activate(cs, feedBackCs);

                    state._predictions[l][p] = _pLayers[l][p]->getHiddenCs();
                }
            }
        }
    }
}

void Hierarchy::writeToStream(
    std::ostream &os
) const {
    int numLayers = _scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        _scLayers[l].writeToStream(os);

        for (int v = 0; v < _pLayers[l].size(); v++) {
            char exists = _pLayers[l][v] != nullptr;

            os.write(reinterpret_cast<const char*>(&exists), sizeof(char));

            if (exists)
                _pLayers[l][v]->writeToStream(os);
        }
    }
}

void Hierarchy::readFromStream(
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;

    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    _inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    _scLayers.resize(numLayers);
    _pLayers.resize(numLayers);

    _historySizes.resize(numLayers);
    
    _ticksPerUpdate.resize(numLayers);

    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        _scLayers[l].readFromStream(is);
        
        _pLayers[l].resize(l == 0 ? _inputSizes.size() : _ticksPerUpdate[l]);

        for (int v = 0; v < _pLayers[l].size(); v++) {
            char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(char));

            if (exists) {
                _pLayers[l][v] = std::make_unique<Predictor>();
                _pLayers[l][v]->readFromStream(is);
            }
            else
                _pLayers[l][v] = nullptr;
        }
    }
}