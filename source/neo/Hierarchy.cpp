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

void Hierarchy::createRandom(ComputeSystem &cs, ComputeProgram &prog,
    const std::vector<cl_int3> &inputSizes, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, std::mt19937 &rng)
{
    _scLayers.resize(layerDescs.size());
    _pLayers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    _updates.resize(layerDescs.size(), false);

    _inputSizes = inputSizes;

    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    for (int l = 0; l < layerDescs.size(); l++) {
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._visibleSize = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
                }
            }
			
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));
			}

            // Predictors
            _pLayers[l].resize(inputSizes.size());

            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(2);

            pVisibleLayerDescs[0]._visibleSize = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;
            
            pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs(2);

            aVisibleLayerDescs[0]._visibleSize = layerDescs[l]._hiddenSize;
            aVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;
            
            aVisibleLayerDescs[1] = aVisibleLayerDescs[0];

            for (int p = 0; p < _pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::_predict) {
                    _pLayers[l][p] = std::make_unique<Predictor>();

                    _pLayers[l][p]->createRandom(cs, prog, inputSizes[p], pVisibleLayerDescs, rng);
                }
                else if (inputTypes[p] == InputType::_act) {
                    // Add an actor
                    _actors[p] = std::make_unique<Actor>();

                    _actors[p]->createRandom(cs, prog, inputSizes[p], aVisibleLayerDescs, rng);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._visibleSize = layerDescs[l - 1]._hiddenSize;
                scVisibleLayerDescs[t]._radius = layerDescs[l]._scRadius;
            }
			
            int inSize = layerDescs[l - 1]._hiddenSize.x * layerDescs[l - 1]._hiddenSize.y;

			for (int v = 0; v < _histories[l].size(); v++) {
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));
            }

            // Predictors
            _pLayers[l].resize(layerDescs[l]._ticksPerUpdate);

            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs(2);

            pVisibleLayerDescs[0]._visibleSize = layerDescs[l]._hiddenSize;
            pVisibleLayerDescs[0]._radius = layerDescs[l]._pRadius;
            
            pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

            for (int p = 0; p < _pLayers[l].size(); p++) {
                _pLayers[l][p] = std::make_unique<Predictor>();

                _pLayers[l][p]->createRandom(cs, prog, layerDescs[l - 1]._hiddenSize, pVisibleLayerDescs, rng);
            }
        }
		
        _scLayers[l].createRandom(cs, prog, layerDescs[l]._hiddenSize, scVisibleLayerDescs, rng);
    }
}

void Hierarchy::step(ComputeSystem &cs, const std::vector<cl::Buffer> &inputCs, const cl::Buffer &topFeedBack, bool learn, float reward) {
    assert(inputCs.size() == _inputSizes.size());

    _ticks[0] = 0;

    // Add to first history   
    {
        int temporalHorizon = _histories.front().size() / _inputSizes.size();

        std::vector<cl::Buffer> lasts(_inputSizes.size());
        
        for (int i = 0; i < _inputSizes.size(); i++)
            lasts[i] = _histories.front()[temporalHorizon - 1 + temporalHorizon * i];
  
        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int i = 0; i < _inputSizes.size(); i++) {
                // Shift
                _histories.front()[t + temporalHorizon * i] = _histories.front()[(t - 1) + temporalHorizon * i];
            }
        }

        for (int i = 0; i < _inputSizes.size(); i++) {
            // Copy
            cs.getQueue().enqueueCopyBuffer(inputCs[i], lasts[i],
                0, 0, _inputSizes[i].x * _inputSizes[i].y * sizeof(cl_int));

            _histories.front()[0 + temporalHorizon * i] = lasts[i];
        }
    }

    _updates.clear();
    _updates.resize(_scLayers.size(), false);

    // Forward
    for (int l = 0; l < _scLayers.size(); l++) {
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            _ticks[l] = 0;

            _updates[l] = true;
            
            _scLayers[l].activate(cs, _histories[l]);

            if (learn)
                _scLayers[l].learn(cs, _histories[l]);

            // Add to next layer's history
            if (l < _scLayers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                cl::Buffer last = _histories[lNext].back();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                // Copy
                cs.getQueue().enqueueCopyBuffer(_scLayers[l].getHiddenCs(), last,
                    0, 0, _scLayers[l].getHiddenSize().x * _scLayers[l].getHiddenSize().y * sizeof(cl_int));
                
                _histories[lNext].front() = last;

                _ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _scLayers.size() - 1; l >= 0; l--) {
        if (_updates[l]) {
            std::vector<cl::Buffer> feedBack(2);

            feedBack[0] = _scLayers[l].getHiddenCs();

            if (l < _scLayers.size() - 1) {
                assert(_pLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]] != nullptr);

                feedBack[1] = _pLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]]->getHiddenCs();
            }
            else
                feedBack[1] = topFeedBack;

            for (int p = 0; p < _pLayers[l].size(); p++) {
                if (_pLayers[l][p] != nullptr) {
                    if (learn) {
                        if (l == 0)
                            _pLayers[l][p]->learn(cs, inputCs[p]);
                        else
                            _pLayers[l][p]->learn(cs, _histories[l][p]);
                    }

                    _pLayers[l][p]->activate(cs, feedBack);
                }
            }

            if (l == 0) {
                for (int p = 0; p < _actors.size(); p++) {
                    if (_actors[p] != nullptr)
                        _actors[p]->step(cs, feedBack, inputCs[p], reward, learn);
                }
            }
        }
    }
}
