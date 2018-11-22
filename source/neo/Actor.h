// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

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
            int _radius;

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
            FloatBuffer _weights;

            Float2 _hiddenToVisible;
            //!@}
        };

        /*!
        \brief History sample
        */
        struct HistorySample {
            std::vector<std::shared_ptr<IntBuffer>> _visibleCs;
            std::shared_ptr<IntBuffer> _hiddenCs;
        
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
        IntBuffer _hiddenCs;

        FloatBuffer _hiddenValues;

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
        void init(int pos, std::mt19937 &rng, int vli);
        void forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputs);
        void learn(const Int2 &pos, std::mt19937 &rng, const std::vector<std::shared_ptr<IntBuffer>> &inputsPrev, const IntBuffer* hiddenCsPrev, float q, float g);

        static void initKernel(int pos, std::mt19937 &rng, Actor* a, int vli) {
            a->init(pos, rng, vli);
        }

        static void forwardKernel(const Int2 &pos, std::mt19937 &rng, Actor* a, const std::vector<const IntBuffer*> &inputs) {
            a->forward(pos, rng, inputs);
        }

        static void learnKernel(const Int2 &pos, std::mt19937 &rng, Actor* a, const std::vector<std::shared_ptr<IntBuffer>> &inputsPrev, const IntBuffer* hiddenCsPrev, float q, float g) {
            a->learn(pos, rng, inputsPrev, hiddenCsPrev, q, g);
        }
        //!@}

    public:
        /*!
        \brief Value learning rate
        */
        float _alpha;

        /*!
        \brief Action learning rate
        */
        float _beta;

        /*!
        \brief Discount factor
        */
        float _gamma;

        /*!
        \brief Initialize defaults
        */
        Actor()
        : _alpha(0.01f), _beta(0.01f), _gamma(0.99f)
        {}

        /*!
        \brief Create an actor layer with random initialization
        \param cs is the ComputeSystem
        \param hiddenSize size of the predictions (output)
        \param historyCapacity maximum number of history samples
        \param visibleLayerDescs are descriptors for visible layers
        \param rng a random number generator
        */
        void createRandom(ComputeSystem &cs,
            Int3 hiddenSize, int historyCapacity, const std::vector<VisibleLayerDesc> &visibleLayerDescs);

        /*!
        \brief Activate the actor (predict values)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        \param rng a random number generator
        \param reward reinforcment signal
        \param learn whether to learn
        */
        void step(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs, float reward, bool learn);

        /*!
        \brief Get number of visible layers
        */
        int getNumVisibleLayers() const {
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
        const IntBuffer &getHiddenCs() const {
            return _hiddenCs;
        }

        /*!
        \brief Get the hidden size
        */
        const Int3 &getHiddenSize() const {
            return _hiddenSize;
        }

        /*!
        \brief Get the weights for a visible layer
        */
        const FloatBuffer &getWeights(int v) {
            return _visibleLayers[v]._weights;
        }
    };
}
