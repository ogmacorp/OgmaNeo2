<!---
  OgmaNeo
  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# OgmaNeo, V2

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby) [![Build Status](https://travis-ci.org/ogmacorp/OgmaNeo.svg?branch=master)](https://travis-ci.org/ogmacorp/OgmaNeo)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) OgmaNeo2 library, C++ library that contains an implementation of Sparse Predictive Hierarchies.

For an introduction to how the algorithm works, see [the presentation](./SPH_Presentation.pdf).
For a more in-depth look, check out [the whitepaper](./OgmaNeo2_Whitepaper_DRAFT.pdf).

There is a [deprecated version](https://github.com/ogmacorp/OgmaNeo) of this repository that contains an outdated implementation of SPH. Please use this version of OgmaNeo (OgmaNeo2, this repository) if possible.

## Installation

### CMake

Version 3.13+ of [CMake](https://cmake.org/) is required when building the library.

### OpenMP

This version of OgmaNeo uses [OpenMP](https://www.openmp.org/) for multiprocessing. It is required in order to build the library. This will typically already be installed on your system.

### Building

The following commands can be used to build the OgmaNeo library:

> git clone https://github.com/ogmacorp/OgmaNeo2.git  
> cd OgmaNeo2
> mkdir build  
> cd build  
> cmake ..  
> make  
> make install

The `cmake` command can be passed a `CMAKE_INSTALL_PREFIX` to determine where to install the library and header files.  

The `BUILD_SHARED_LIBS` boolean cmake option can be used to create dynamic/shared object library (default is to create a _static_ library). On Linux it's recommended to add `-DBUILD_SHARED_LIBS=ON` (especially if you plan to use the Python bindings in PyOgmaNeo2).

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On **Windows** systems it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters, such as `CMAKE_INSTALL_PREFIX`.

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to OgmaNeo2.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [OGMANEO_LICENSE.md](./OGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeo Copyright (c) 2016-2020 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.