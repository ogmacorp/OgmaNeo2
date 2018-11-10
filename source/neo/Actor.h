// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace ogmaneo {
    /*!
    \brief A 2D prediction layer
    Predicts the targets one timestep ahead of time
    */
    class Actor {
    public:
        /*!
        \brief Visible layer descriptor
        */
        struct VisibleLayerDesc {
            /*!
            \brief Visible layer size
            */
            Int3 _size;

            /*!
            \brief Radius onto hidden layer
            */
            cl_int _radius;
            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 4, 4, 16 }),
                _radius(2)
            {}
        };

        /*!
        \brief Visible layer
        */
        struct VisibleLayer {
            //!@{
            /*!
            \brief Visible layer values and buffers
            */
            cl::Buffer _weights;

            Float2 _hiddenToVisible;
            //!@}
        };

        /*!
        \brief History sample
        */
        struct HistorySample {
            std::vector<cl::Buffer> _visibleCs;
            cl::Buffer _hiddenCs;
        
            float _reward;
        };

    private:
        /*!
        \brief Size of the hidden layer (output)
        */
        Int3 _hiddenSize;

        /*!
        \brief Current history size
        */
        int _historySize;

        //!@{
        /*!
        \brief Buffers
        */
        cl::Buffer _hiddenCs;
        DoubleBuffer _hiddenActivations;

        std::vector<HistorySample> _historySamples;
        //!@}

        //!@{
        /*!
        \brief Visible layers and associated descriptors
        */
        std::vector<VisibleLayer> _visibleLayers;
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        //!@}

        //!@{
        /*!
        \brief Kernels
        */
        cl::Kernel _forwardKernel;
        cl::Kernel _inhibitKernel;
        cl::Kernel _learnKernel;
        //!@}

    public:
        /*!
        \brief Value learning rate
        */
        cl_float _alpha;

        /*!
        \brief Action learning rate
        */
        cl_float _beta;

        /*!
        \brief Discount factor
        */
        cl_float _gamma;

        /*!
        \brief Exploration rate
        */
        cl_float _epsilon;

        /*!
        \brief Initialize defaults
        */
        Actor()
        : _alpha(0.01f), _beta(0.1f), _gamma(0.98f), _epsilon(0.01f)
        {}

        /*!
        \brief Create an actor layer with random initialization
        \param cs is the ComputeSystem
        \param prog is the ComputeProgram associated with the ComputeSystem and loaded with the actor kernel code
        \param hiddenSize size of the predictions (output)
        \param historyCapacity maximum number of history samples
        \param visibleLayerDescs are descriptors for visible layers
        \param rng a random number generator
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &prog,
            Int3 hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            std::mt19937 &rng);

        /*!
        \brief Activate the actor (predict values)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        \param rng a random number generator
        \param reward reinforcment signal
        \param learn whether to learn
        */
        void step(ComputeSystem &cs, const std::vector<cl::Buffer> &visibleCs, std::mt19937 &rng, float reward, bool learn);

        /*!
        \brief Write to stream.
        */
        void writeToStream(ComputeSystem &cs, std::ostream &os);

        /*!
        \brief Read from stream (create).
        */
        void readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is); 

        /*!
        \brief Get number of visible layers
        */
        size_t getNumVisibleLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get a visible layer
        */
        const VisibleLayer &getVisibleLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get a visible layer descriptor
        */
        const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get the hidden activations (predictions)
        */
        const cl::Buffer &getHiddenCs() const {
            return _hiddenCs;
        }

        /*!
        \brief Get the hidden size
        */
        Int3 getHiddenSize() const {
            return _hiddenSize;
        }

        /*!
        \brief Get the weights for a visible layer
        */
        const cl::Buffer &getWeights(int v) {
            return _visibleLayers[v]._weights;
        }
    };
}
