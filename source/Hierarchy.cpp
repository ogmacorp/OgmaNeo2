// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace ogmaneo;

void Hierarchy::init(
    ComputeSystem &cs,
    ComputeProgram &prog,
    const std::vector<Int3> &inputSizes,
    const std::vector<InputType> &inputTypes,
    const std::vector<LayerDesc> &layerDescs,
    std::mt19937 &rng
) {
    _scLayers.resize(layerDescs.size());
    _aLayers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    _historySizes.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    _rewards.resize(layerDescs.size(), 0.0f);
    _rewardCounts = _rewards;

    _updates.resize(layerDescs.size(), false);

    _inputSizes = inputSizes;

    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    for (int l = 0; l < layerDescs.size(); l++) {
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        _historySizes[l].resize(_histories[l].size());
			
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = t + layerDescs[l]._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = layerDescs[l]._scRadius;
                }
            }
            
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / layerDescs[l]._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));

                _historySizes[l][v] = inSize;
			}

            // Actors
            _aLayers[l].resize(inputSizes.size());

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

            for (int p = 0; p < _aLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::_act) {
                    _aLayers[l][p] = std::make_unique<Actor>();

                    _aLayers[l][p]->init(cs, prog, inputSizes[p], layerDescs[l]._historyCapacity, aVisibleLayerDescs, rng);
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
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));

                _historySizes[l][v] = inSize;
            }

            // Actors
            _aLayers[l].resize(layerDescs[l]._ticksPerUpdate);

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

            for (int p = 0; p < _aLayers[l].size(); p++) {
                _aLayers[l][p] = std::make_unique<Actor>();

                _aLayers[l][p]->init(cs, prog, layerDescs[l - 1]._hiddenSize, layerDescs[l]._historyCapacity, aVisibleLayerDescs, rng);
            }
        }
		
        _scLayers[l].init(cs, prog, layerDescs[l]._hiddenSize, scVisibleLayerDescs, rng);
    }
}

void Hierarchy::step(
    ComputeSystem &cs,
    const std::vector<cl::Buffer> &inputCs,
    std::mt19937 &rng,
    float reward,
    bool learnEnabled
) {
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
        _rewards[l] += reward;
        _rewardCounts[l] += 1.0f;

        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            _ticks[l] = 0;

            _updates[l] = true;
            
            _scLayers[l].step(cs, _histories[l], learnEnabled);

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
            std::vector<cl::Buffer> feedBack(l < _scLayers.size() - 1 ? 2 : 1);

            feedBack[0] = _scLayers[l].getHiddenCs();

            if (l < _scLayers.size() - 1) {
                assert(_aLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]] != nullptr);

                feedBack[1] = _aLayers[l + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]]->getHiddenCs();
            }

            float r = _rewards[l] / std::max(1.0f, _rewardCounts[l]);

            _rewards[l] = 0.0f;
            _rewardCounts[l] = 0.0f;

            for (int p = 0; p < _aLayers[l].size(); p++) {
                if (_aLayers[l][p] != nullptr) {
                    _aLayers[l][p]->step(cs, feedBack, rng, r, learnEnabled);
                }
            }
        }
    }
}

void Hierarchy::writeToStream(
    ComputeSystem &cs,
    std::ostream &os
) {
    int numLayers = _scLayers.size();

    os.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<const char*>(&numInputs), sizeof(int));

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(cl_int3));

    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(unsigned char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    os.write(reinterpret_cast<const char*>(_rewards.data()), _rewards.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(_rewardCounts.data()), _rewardCounts.size() * sizeof(float));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++) {
            std::vector<cl_int> historyCs(_historySizes[l][i]);
            cs.getQueue().enqueueReadBuffer(_histories[l][i], CL_TRUE, 0, _historySizes[l][i] * sizeof(cl_int), historyCs.data());
            os.write(reinterpret_cast<const char*>(historyCs.data()), _historySizes[l][i] * sizeof(cl_int));
        }

        _scLayers[l].writeToStream(cs, os);

        for (int v = 0; v < _aLayers[l].size(); v++) {
            unsigned char exists = _aLayers[l][v] != nullptr;

            os.write(reinterpret_cast<const char*>(&exists), sizeof(unsigned char));

            if (exists)
                _aLayers[l][v]->writeToStream(cs, os);
        }
    }
}

void Hierarchy::readFromStream(
    ComputeSystem &cs,
    ComputeProgram &prog,
    std::istream &is
) {
    int numLayers;
    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));

    int numInputs;
    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));
    _inputSizes.resize(numInputs);
    is.read(reinterpret_cast<char*>(_inputSizes.data()), numInputs * sizeof(cl_int3));

    _scLayers.resize(numLayers);
    _aLayers.resize(numLayers);

    _ticks.resize(numLayers);

    _histories.resize(numLayers);
    _historySizes.resize(numLayers);
    
    _ticksPerUpdate.resize(numLayers);

    _updates.resize(numLayers);

    _rewards.resize(numLayers);
    _rewardCounts.resize(numLayers);

    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(unsigned char));
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    is.read(reinterpret_cast<char*>(_rewards.data()), _rewards.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(_rewardCounts.data()), _rewardCounts.size() * sizeof(float));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        _historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        _histories[l].resize(numHistorySizes);

        for (int i = 0; i < _historySizes[l].size(); i++) {
            std::vector<cl_int> historyCs(_historySizes[l][i]);
            is.read(reinterpret_cast<char*>(historyCs.data()), _historySizes[l][i] * sizeof(cl_int));
            _histories[l][i] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _historySizes[l][i] * sizeof(cl_int));
            cs.getQueue().enqueueWriteBuffer(_histories[l][i], CL_TRUE, 0, _historySizes[l][i] * sizeof(cl_int), historyCs.data());   
        }

        _scLayers[l].readFromStream(cs, prog, is);

        _aLayers[l].resize(l == 0 ? _inputSizes.size() : _ticksPerUpdate[l]);

        for (int v = 0; v < _aLayers[l].size(); v++) {
            unsigned char exists;

            is.read(reinterpret_cast<char*>(&exists), sizeof(unsigned char));

            if (exists) {
                _aLayers[l][v] = std::make_unique<Actor>();
                _aLayers[l][v]->readFromStream(cs, prog, is);
            }
            else
                _aLayers[l][v] = nullptr;
        }
    }
}