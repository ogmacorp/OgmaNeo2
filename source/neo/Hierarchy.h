// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"
#include "Predictor.h"
#include "Actor.h"

#include <memory>

namespace ogmaneo {
    /*!
    \brief Enum describing the type of operation performed by an input layer
    */
    enum InputType {
        _none, _predict, _act
    };
    
    /*!
    \brief A hierarchy of sparse coders and predictors, using the exponential memory structure
    */
    class Hierarchy {
    public:
        /*!
        \brief Parameters for a layer
        Used during construction of a hierarchy
        */
        struct LayerDesc {
            /*!
            \brief Dimensions of the hidden layer
            */
            Int3 _hiddenSize;

            //!@{
            /*!
            \brief Radii of the sparse coder and predictor/actor
            */
            cl_int _scRadius;
            cl_int _aRadius;
            //!@}

            /*!
            \brief Number of ticks a layer takes to update (relative to previous layer)
            */
            int _ticksPerUpdate;

            /*!
            \brief Temporal distance into a the past addressed by the layer. Should be greater than or equal to _ticksPerUpdate
            */
            int _temporalHorizon;

            /*!
            \brief History capacity
            */
            int _historyCapacity;

            /*!
            \brief Initialize defaults
            */
            LayerDesc()
                : _hiddenSize(4, 4, 16),
                _scRadius(2), _aRadius(2),
                _ticksPerUpdate(2), _temporalHorizon(2),
                _historyCapacity(8)
            {}
        };
    private:
        std::vector<SparseCoder> _scLayers;
        std::vector<std::vector<std::unique_ptr<Actor>>> _aLayers;

        std::vector<std::vector<cl::Buffer>> _histories;
        std::vector<std::vector<int>> _historySizes;

        std::vector<char> _updates;

        std::vector<float> _rewards;
        std::vector<float> _rewardCounts;

        std::vector<int> _ticks;
        std::vector<int> _ticksPerUpdate;

        std::vector<Int3> _inputSizes;

    public:
        /*!
        \brief Create a randomly initialized hierarchy
        \param cs is the ComputeSystem
        \param prog is the ComputeProgram associated with the ComputeSystem and loaded with the sparse coder and predictor kernel code
        \param inputSizes vector of input dimensions
        \param predictInputs flags for which inputs to generate predictions for
        \param layerDescs vector of LayerDesc structures, describing each layer in sequence
        \param rng a random number generator
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &prog,
            const std::vector<Int3> &inputSizes, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, std::mt19937 &rng);

        /*!
        \brief Simulation step/tick
        \param inputs vector of input activations
        \param learn whether learning should be enabled, defaults to true
        */
        void step(ComputeSystem &cs, const std::vector<cl::Buffer> &inputCs, std::mt19937 &rng, bool learn = true, float reward = 0.0f);

        /*!
        \brief Write to stream.
        */
        void writeToStream(ComputeSystem &cs, std::ostream &os);

        /*!
        \brief Read from stream (create).
        */
        void readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is);

        /*!
        \brief Get the number of (hidden) layers
        */
        int getNumLayers() const {
            return _scLayers.size();
        }

        /*!
        \brief Get the actor output
        \param i the index of the input to retrieve
        */
        const cl::Buffer &getActionCs(int i) const {
            return _aLayers.front()[i]->getHiddenCs();
        }

        /*!
        \brief Whether this layer received on update this timestep
        */
        bool getUpdate(int l) const {
            return _updates[l];
        }

        /*!
        \brief Get current layer ticks, relative to previous layer
        */
        int getTicks(int l) const {
            return _ticks[l];
        }

        /*!
        \brief Get layer ticks per update, relative to previous layer
        */
        int getTicksPerUpdate(int l) const {
            return _ticksPerUpdate[l];
        }

        /*!
        \brief Get input sizes.
        */
        const std::vector<Int3> &getInputSizes() const {
            return _inputSizes;
        }

        /*!
        \brief Retrieve a sparse coding layer
        */
        SparseCoder &getSCLayer(int l) {
            return _scLayers[l];
        }

        /*!
        \brief Retrieve predictor layer(s)
        */
        std::vector<std::unique_ptr<Actor>> &getALayer(int l) {
            return _aLayers[l];
        }
    };
}
