<!---
  OgmaNeo
  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# OgmaNeo, V2 - goaltron branch

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) OgmaNeo2 library, C++ library that contains an implementation of Sparse Predictive Hierarchies (SPH).

This particular branch contains a CPU-only implementation of the goaltron, and version of SPH capable of performing reinforcement-learning-like tasks without the use of temporal difference learning.

There is also a [deprecated version](https://github.com/ogmacorp/OgmaNeo) of this repository that contains an outdated implementation of SPH. Please use this version of OgmaNeo (OgmaNeo2, this repository) if possible.

## Installation

### CMake

Version 3.8+ of [CMake](https://cmake.org/) is required when building the library.

### Building

The following commands can be used to build the OgmaNeo library:

> git clone https://github.com/ogmacorp/OgmaNeo2.git  
> cd OgmaNeo2
> mkdir build  
> cd build  
> cmake -DCMAKE_INSTALL_PREFIX=../install ..  
> make  
> make install

The `cmake` command can be passed a `CMAKE_INSTALL_PREFIX` to determine where to install the library and header files.  

The `BUILD_SHARED_LIBS` boolean cmake option can be used to create dynamic/shared object library (default is to create a _static_ library). On Linux it's recommended to add `-DBUILD_SHARED_LIBS=ON`

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On **Windows** systems it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters, such as `CMAKE_INSTALL_PREFIX`.

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to OgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [OGMANEO_LICENSE.md](./OGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeo Copyright (c) 2016-2018 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.

Note: External libraries used:
CTPL (for thread pooling): https://github.com/vit-vit/CTPL