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

void Hierarchy::createRandom(ComputeSystem &cs,
    const std::vector<Int3> &inputSizes, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs)
{
    // Create layers
    _scLayers.resize(layerDescs.size());
    _aLayers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    _updates.resize(layerDescs.size(), false);

    // Set reward accumulators to 0
    _rewards.resize(layerDescs.size(), 0.0f);
    _rewardCounts = _rewards;

    // Cache input sizes
    _inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Histories for all input layers or just the one sparse coder (if not the first layer)
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        _historySizes[l].resize(_histories[l].size());
		
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
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = std::make_shared<IntBuffer>(inSize);

#ifdef KERNEL_DEBUG
                for (int x = 0; x < inSize; x++)
                    fillInt(x, cs._rng, _histories[l][v].get(), 0);
#else
                runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, _histories[l][v].get(), 0), inSize, cs._rng, cs._batchSize1);
#endif

                _historySizes[l][v] = inSize;
			}

            // Predictors
            _aLayers[l].resize(inputSizes.size());

            // Create actor visible layer descriptors
            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs;

            if (l < layerDescs.size() - 1) {
                aVisibleLayerDescs.resize(2);

                aVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
                aVisibleLayerDescs[0]._radius = layerDescs[l]._aRadius;

                aVisibleLayerDescs[1] = aVisibleLayerDescs[0];
            }
            else {
                aVisibleLayerDescs.resize(1);

                aVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
                aVisibleLayerDescs[0]._radius = layerDescs[l]._aRadius;
            }

            // Create actors
            for (int p = 0; p < _aLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::_act) {
                    _aLayers[l][p] = std::make_unique<Actor>();

                    _aLayers[l][p]->createRandom(cs, inputSizes[p], layerDescs[l]._historyCapacity, aVisibleLayerDescs);
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

			for (int v = 0; v < _histories[l].size(); v++) {
                _histories[l][v] = std::make_shared<IntBuffer>(inSize);

#ifdef KERNEL_DEBUG
                for (int x = 0; x < inSize; x++)
                    fillInt(x, cs._rng, _histories[l][v].get(), 0);
#else
                runKernel1(cs, std::bind(fillInt, std::placeholders::_1, std::placeholders::_2, _histories[l][v].get(), 0), inSize, cs._rng, cs._batchSize1);
#endif

                _historySizes[l][v] = inSize;
            }

            _aLayers[l].resize(layerDescs[l]._ticksPerUpdate);

            // Actor visible layer descriptors
            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs;

            if (l < layerDescs.size() - 1) {
                aVisibleLayerDescs.resize(2);

                aVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
                aVisibleLayerDescs[0]._radius = layerDescs[l]._aRadius;

                aVisibleLayerDescs[1] = aVisibleLayerDescs[0];
            }
            else {
                aVisibleLayerDescs.resize(1);
                
                aVisibleLayerDescs[0]._size = layerDescs[l]._hiddenSize;
                aVisibleLayerDescs[0]._radius = layerDescs[l]._aRadius;
            }

            // Create actors
            for (int p = 0; p < _aLayers[l].size(); p++) {
                _aLayers[l][p] = std::make_unique<Actor>();

                _aLayers[l][p]->createRandom(cs, layerDescs[l - 1]._hiddenSize, layerDescs[l]._historyCapacity, aVisibleLayerDescs);
            }
        }
		
        // Create the sparse coding layer
        _scLayers[l].createRandom(cs, layerDescs[l]._hiddenSize, scVisibleLayerDescs);
    }
}

void Hierarchy::step(ComputeSystem &cs, const std::vector<const IntBuffer*> &inputCs, bool learnEnabled, float reward) {
    assert(inputCs.size() == _inputSizes.size());

    // First tick is always 0
    _ticks[0] = 0;

    // Add input to first layer history   
    {
        int temporalHorizon = _histories.front().size() / _inputSizes.size();

        std::vector<std::shared_ptr<IntBuffer>> lasts(_inputSizes.size());
        
        for (int i = 0; i < _inputSizes.size(); i++)
            lasts[i] = _histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < _inputSizes.size(); i++) {
                // Shift
                _histories.front()[t + temporalHorizon * i] = _histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < _inputSizes.size(); i++) {
            assert(_inputSizes[i].x * _inputSizes[i].y == inputCs[i]->size());
            
            // Copy
#ifdef KERNEL_DEBUG
            for (int x = 0; x < inputCs[i]->size(); x++)
                copyInt(x, cs._rng, inputCs[i], lasts[i].get());
#else
            runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, inputCs[i], lasts[i].get()), inputCs[i]->size(), cs._rng, cs._batchSize1);
#endif

            _histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    // Set all updates to no update, will be set to true if an update occurred later
    _updates.clear();
    _updates.resize(_scLayers.size(), false);

    // Forward
    for (int l = 0; l < _scLayers.size(); l++) {
        // Accumulate reward
        _rewards[l] += reward;
        _rewardCounts[l] += 1.0f;

        // If is time for layer to tick
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            // Reset tick
            _ticks[l] = 0;

            // Updated
            _updates[l] = true;
            
            // Activate sparse coder
            _scLayers[l].activate(cs, constGet(_histories[l]));

            // Optionally learn sparse coding layer
            if (learnEnabled)
                _scLayers[l].learn(cs, constGet(_histories[l]));

            // Add to next layer's history
            if (l < _scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                std::shared_ptr<IntBuffer> last = _histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                // Copy
#ifdef KERNEL_DEBUG
                for (int x = 0; x < _scLayers[l].getHiddenCs().size(); x++)
                    copyInt(x, cs._rng, &_scLayers[l].getHiddenCs(), last.get());
#else
                runKernel1(cs, std::bind(copyInt, std::placeholders::_1, std::placeholders::_2, &_scLayers[l].getHiddenCs(), last.get()), _scLayers[l].getHiddenCs().size(), cs._rng, cs._batchSize1);
#endif

                _histories[lNext].front() = last;

                _ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _scLayers.size() - 1; l >= 0; l--) {
        if (_updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            std::vector<const IntBuffer*> feedBack(l < _scLayers.size() - 1 ? 2 : 1);

            feedBack[0] = &_scLayers[l].getHiddenCs();

            if (l < _scLayers.size() - 1) {
                assert(_aLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]] != nullptr);

                feedBack[1] = &_aLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]]->getHiddenCs();
            }

            // Determine reward from accumulators
            float r = _rewards[l] / std::max(1.0f, _rewardCounts[l]);

            // Reset accumulators
            _rewards[l] = 0.0f;
            _rewardCounts[l] = 0.0f;

            // Step actor layers
            for (int p = 0; p < _aLayers[l].size(); p++) {
                if (_aLayers[l][p] != nullptr)
                    _aLayers[l][p]->step(cs, feedBack, l == 0 ? inputCs[p] : _histories[l][p].get(), r, learnEnabled);
            }
        }
    }
}