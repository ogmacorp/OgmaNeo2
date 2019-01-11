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
    \brief Sparse Coder
    A 2D sparse coding layer, using Columnar Binary Sparse Coding (computes CSDR -> compressed CSDR)
    */
    class ImageEncoder {
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

            FloatBuffer _visibleActivations; // For reconstruction

            Float2 _visibleToHidden; // For projection
            Float2 _hiddenToVisible; // For projection

            Int2 _reverseRadii; // Pre-computed reverse radii
            //!@}
        };

    private:
        /*!
        \brief Size of the hidden layer
        */
        Int3 _hiddenSize;

        /*!
        \brief Buffers for hidden state
        */
        IntBuffer _hiddenCs;

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
        void forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const FloatBuffer*> &inputActivations, bool learnEnabled);
        void backward(const Int2 &pos, std::mt19937 &rng, const IntBuffer* hiddenCs, int vli);

        static void initKernel(int pos, std::mt19937 &rng, ImageEncoder* sc, int vli) {
            sc->init(pos, rng, vli);
        }

        static void forwardKernel(const Int2 &pos, std::mt19937 &rng, ImageEncoder* sc, const std::vector<const FloatBuffer*> &inputActivations, bool learnEnabled) {
            sc->forward(pos, rng, inputActivations, learnEnabled);
        }

        static void backwardKernel(const Int2 &pos, std::mt19937 &rng, ImageEncoder* sc, const IntBuffer* hiddenCs, int vli) {
            sc->backward(pos, rng, hiddenCs, vli);
        }
        //!@}

    public:
        /*!
        \brief Learning rate
        */
        float _alpha;

        /*!
        \brief Initialize defaults
        */
        ImageEncoder()
        : _alpha(0.1f)
        {}

        /*!
        \brief Create a sparse coding layer with random initialization
        \param cs is the ComputeSystem
        \param hiddenSize size of the hidden layer
        \param visibleLayerDescs the descriptors for the visible layers
        */
        void createRandom(ComputeSystem &cs,
            const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs);

        /*!
        \brief Activate the sparse coder (perform sparse coding)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        */
        void step(ComputeSystem &cs, const std::vector<const FloatBuffer*> &inputActivations, bool learnEnabled);

        /*!
        \brief Reconstruction of input data from hidden CSDR
        */
        void reconstruct(ComputeSystem &cs, const IntBuffer* hiddenCs);

        /*!
        \brief Write to stream
        */
        void writeToStream(std::ostream &os) const;

        /*!
        \brief Read from stream
        */
        void readFromStream(std::istream &is);
        
        /*!
        \brief Get the number of visible layers
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
        \brief Get the hidden activations (state)
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
        \brief Get a visible layer's feed weights
        */
        const FloatBuffer &getWeights(int v) const {
            return _visibleLayers[v]._weights;
        }
    };
}
