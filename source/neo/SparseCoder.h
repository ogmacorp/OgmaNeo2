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
    class SparseCoder {
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

            FloatBuffer _visibleActivations;

<<<<<<< HEAD
            Float2 _visibleToHidden; // For projection
            Float2 _hiddenToVisible; // For projection

            Int2 _reverseRadii; // Pre-computed reverse radii
=======
            Float2 _visibleToHidden;
            Float2 _hiddenToVisible;

            Int2 _reverseRadii;
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
            //!@}
        };

    private:
        /*!
        \brief Size of the hidden layer
        */
        Int3 _hiddenSize;

        //!@{
        /*!
        \brief Buffers for hidden state
        */
        IntBuffer _hiddenCs;
        IntBuffer _hiddenCsPrev;

        FloatBuffer _hiddenActivations;
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
<<<<<<< HEAD
        void init(int pos, std::mt19937 &rng, int vli);
        void forward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, bool firstIter);
        void backward(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli);
        void learn(const Int2 &pos, std::mt19937 &rng, const std::vector<const IntBuffer*> &inputCs, int vli);

        static void initKernel(int pos, std::mt19937 &rng, SparseCoder* sc, int vli) {
            sc->init(pos, rng, vli);
        }

        static void forwardKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputCs, bool firstIter) {
            sc->forward(pos, rng, inputCs, firstIter);
        }

        static void backwardKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputCs, int vli) {
            sc->backward(pos, rng, inputCs, vli);
        }

        static void learnKernel(const Int2 &pos, std::mt19937 &rng, SparseCoder* sc, const std::vector<const IntBuffer*> &inputCs, int vli) {
            sc->learn(pos, rng, inputCs, vli);
        }
=======
        cl::Kernel _forwardKernel;
        cl::Kernel _backwardPartialKernel;
        cl::Kernel _backwardKernel;
        cl::Kernel _inhibitKernel;
        cl::Kernel _learnKernel;
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
        //!@}

    public:
        /*!
        \brief Learning rate
        */
        float _alpha;

        /*!
        \brief Explaining-away iterations (part of iterative sparse coding)
        */
        int _explainIters;

        /*!
        \brief Initialize defaults
        */
        SparseCoder()
        : _alpha(0.001f), _explainIters(4)
        {}

        /*!
        \brief Create a sparse coding layer with random initialization
        \param cs is the ComputeSystem
        \param hiddenSize size of the hidden layer
        \param visibleLayerDescs the descriptors for the visible layers
        */
<<<<<<< HEAD
        void createRandom(ComputeSystem &cs,
            const Int3 &hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs);
=======
        void createRandom(ComputeSystem &cs, ComputeProgram &prog,
            Int3 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            std::mt19937 &rng);
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e

        /*!
        \brief Activate the sparse coder (perform sparse coding)
        \param cs is the ComputeSystem
        \param visibleCs the visible (input) layer states
        */
        void activate(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs);

        /*!
        \brief Learn the sparse code
        \param cs is the ComputeSystem.
        \param visibleCs the visible (input) layer states
        */
        void learn(ComputeSystem &cs, const std::vector<const IntBuffer*> &visibleCs);

        /*!
        \brief Write to stream.
        */
        void writeToStream(ComputeSystem &cs, std::ostream &os);

        /*!
        \brief Read from stream (create).
        */
        void readFromStream(ComputeSystem &cs, ComputeProgram &prog, std::istream &is); 

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
        \brief Get the hidden activations (state)
        */
        const IntBuffer &getHiddenCsPrev() const {
            return _hiddenCsPrev;
        }

        /*!
        \brief Get the hidden size
        */
<<<<<<< HEAD
        const Int3 &getHiddenSize() const {
=======
        Int3 getHiddenSize() const {
>>>>>>> 4fa97ae0f684e2beabb2f68b1994bbe2033fa71e
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
