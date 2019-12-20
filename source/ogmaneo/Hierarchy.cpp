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
    const FirstLayerDesc &firstLayerDesc,
    const std::vector<HigherLayerDesc> &higherLayerDescs,
    std::mt19937 &rng
) {
    _scLayers.resize(higherLayerDescs.size() + 1);
    _pLayers.resize(higherLayerDescs.size());

    _ticks.assign(_scLayers.size(), 0);

    _histories.resize(_scLayers.size());
    _historySizes.resize(_scLayers.size());
    
    _ticksPerUpdate.resize(_scLayers.size());

    _updates.resize(_scLayers.size(), false);

    _inputSizes = inputSizes;

    for (int l = 0; l < _scLayers.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : higherLayerDescs[l - 1]._ticksPerUpdate; // First layer always 1

    for (int l = 0; l < _scLayers.size(); l++) {
        int pl = l - 1;

        _histories[l].resize(l == 0 ? inputSizes.size() * firstLayerDesc._temporalHorizon : higherLayerDescs[pl]._temporalHorizon);

        _historySizes[l].resize(_histories[l].size());
			
        std::vector<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * firstLayerDesc._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < firstLayerDesc._temporalHorizon; t++) {
                    int index = t + firstLayerDesc._temporalHorizon * i;

                    scVisibleLayerDescs[index]._size = inputSizes[i];
                    scVisibleLayerDescs[index]._radius = firstLayerDesc._ffRadius;
                }
            }
            
			for (int v = 0; v < _histories[l].size(); v++) {
				int i = v / firstLayerDesc._temporalHorizon;

                int inSize = inputSizes[i].x * inputSizes[i].y;
				
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));

                _historySizes[l][v] = inSize;
			}

            // Actors
            _aLayers.resize(inputSizes.size());

            std::vector<Actor::VisibleLayerDesc> aVisibleLayerDescs;

            if (l < _scLayers.size() - 1) {
                aVisibleLayerDescs.resize(2);

                aVisibleLayerDescs[0]._size = firstLayerDesc._hiddenSize;
                aVisibleLayerDescs[0]._radius = firstLayerDesc._aRadius;

                aVisibleLayerDescs[1] = aVisibleLayerDescs[0];
            }
            else {
                aVisibleLayerDescs.resize(1);

                aVisibleLayerDescs[0]._size = firstLayerDesc._hiddenSize;
                aVisibleLayerDescs[0]._radius = firstLayerDesc._aRadius;
            }

            for (int a = 0; a < _aLayers.size(); a++) {
                if (inputTypes[a] == InputType::_act) {
                    _aLayers[a] = std::make_unique<Actor>();

                    _aLayers[a]->init(cs, prog, inputSizes[a], firstLayerDesc._historyCapacity, aVisibleLayerDescs, rng);
                }
                else
                    _aLayers[a] = nullptr;
            }
        }
        else {
            scVisibleLayerDescs.resize(higherLayerDescs[pl]._temporalHorizon);

            for (int t = 0; t < higherLayerDescs[pl]._temporalHorizon; t++) {
                scVisibleLayerDescs[t]._size = _scLayers[l - 1].getHiddenSize();
                scVisibleLayerDescs[t]._radius = higherLayerDescs[pl]._ffRadius;
            }

            int inSize = _scLayers[l - 1].getHiddenSize().x * _scLayers[l - 1].getHiddenSize().y;

			for (int v = 0; v < _histories[l].size(); v++) {
				_histories[l][v] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, inSize * sizeof(cl_int));

                cs.getQueue().enqueueFillBuffer(_histories[l][v], static_cast<cl_int>(0), 0, inSize * sizeof(cl_int));

                _historySizes[l][v] = inSize;
            }

            // Predictors
            _pLayers[pl].resize(higherLayerDescs[pl]._ticksPerUpdate);

            std::vector<Predictor::VisibleLayerDesc> pVisibleLayerDescs;

            if (l < _scLayers.size() - 1) {
                pVisibleLayerDescs.resize(2);

                pVisibleLayerDescs[0]._size = higherLayerDescs[pl]._hiddenSize;
                pVisibleLayerDescs[0]._radius = higherLayerDescs[pl]._pRadius;

                pVisibleLayerDescs[1] = pVisibleLayerDescs[0];
            }
            else {
                pVisibleLayerDescs.resize(1);
                
                pVisibleLayerDescs[0]._size = higherLayerDescs[pl]._hiddenSize;
                pVisibleLayerDescs[0]._radius = higherLayerDescs[pl]._pRadius;
            }

            for (int p = 0; p < _pLayers[pl].size(); p++)
                _pLayers[pl][p].init(cs, prog, _scLayers[l - 1].getHiddenSize(), pVisibleLayerDescs, rng);
        }
		
        _scLayers[l].init(cs, prog, l == 0 ? firstLayerDesc._hiddenSize : higherLayerDescs[pl]._hiddenSize, scVisibleLayerDescs, rng);
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
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            _ticks[l] = 0;

            _updates[l] = true;
            
            _scLayers[l].step(cs, _histories[l], rng, learnEnabled);

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
            int pl = l - 1;

            std::vector<cl::Buffer> feedBack(l < _scLayers.size() - 1 ? 2 : 1);

            feedBack[0] = _scLayers[l].getHiddenCs();

            if (l < _scLayers.size() - 1)
                feedBack[1] = _pLayers[pl + 1][_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]].getHiddenCs();

            if (l == 0) {
                for (int a = 0; a < _aLayers.size(); a++) {
                    if (_aLayers[a] != nullptr)
                        _aLayers[a]->step(cs, feedBack, rng, reward, learnEnabled);
                }
            }
            else {
                for (int p = 0; p < _pLayers[pl].size(); p++)
                    _pLayers[pl][p].step(cs, feedBack, _histories[l][p], learnEnabled);
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

    os.write(reinterpret_cast<const char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    os.write(reinterpret_cast<const char*>(_updates.data()), _updates.size() * sizeof(unsigned char));
    os.write(reinterpret_cast<const char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<const char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes = _historySizes[l].size();

        os.write(reinterpret_cast<const char*>(&numHistorySizes), sizeof(int));

        os.write(reinterpret_cast<const char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        for (int i = 0; i < _historySizes[l].size(); i++)
            writeBufferToStream(cs, os, _histories[l][i], _historySizes[l][i] * sizeof(cl_int));

        _scLayers[l].writeToStream(cs, os);

        if (l == 0) {
            for (int a = 0; a < _aLayers.size(); a++) {
                unsigned char exists = _aLayers[a] != nullptr;

                os.write(reinterpret_cast<const char*>(&exists), sizeof(unsigned char));

                if (exists)
                    _aLayers[a]->writeToStream(cs, os);
            }
        }
        else {
            int pl = l - 1;

            for (int p = 0; p < _pLayers[pl].size(); p++)
                _pLayers[pl][p].writeToStream(cs, os);
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
    is.read(reinterpret_cast<char*>(_inputSizes.data()), numInputs * sizeof(Int3));

    _scLayers.resize(numLayers);
    _pLayers.resize(numLayers - 1);
    _aLayers.resize(numInputs);

    _ticks.resize(numLayers);

    _histories.resize(numLayers);
    _historySizes.resize(numLayers);
    
    _ticksPerUpdate.resize(numLayers);

    _updates.resize(numLayers);

    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(unsigned char));
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numHistorySizes;
        
        is.read(reinterpret_cast<char*>(&numHistorySizes), sizeof(int));
        _historySizes[l].resize(numHistorySizes);
        is.read(reinterpret_cast<char*>(_historySizes[l].data()), numHistorySizes * sizeof(int));

        _histories[l].resize(numHistorySizes);

        for (int i = 0; i < _historySizes[l].size(); i++)
            readBufferFromStream(cs, is, _histories[l][i], _historySizes[l][i] * sizeof(cl_int));

        _scLayers[l].readFromStream(cs, prog, is);

        if (l == 0) {
            for (int a = 0; a < _aLayers.size(); a++) {
                unsigned char exists;

                is.read(reinterpret_cast<char*>(&exists), sizeof(unsigned char));

                if (exists) {
                    _aLayers[a] = std::make_unique<Actor>();
                    _aLayers[a]->readFromStream(cs, prog, is);
                }
                else
                    _aLayers[a] = nullptr;
            }
        }
        else {
            int pl = l - 1;

            _pLayers[pl].resize(_ticksPerUpdate[l]);

            for (int p = 0; p < _pLayers[pl].size(); p++)
                _pLayers[pl][p].readFromStream(cs, prog, is);
        }
    }
}