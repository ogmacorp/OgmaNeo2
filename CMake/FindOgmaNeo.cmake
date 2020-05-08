# --------------------------------------------------------------------------
# OgmaNeo
# Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeo is licensed to you under the terms described
# in the OGMANEO_LICENSE.md file included in this distribution.
# --------------------------------------------------------------------------

# Locate OgmaNeo library
#
# This module defines
# OGMANEO_LIBRARY, the name of the library to link against
# OGMANEO_FOUND, if false, do not try to link to OgmaNeo
# OGMANEO_INCLUDE_DIR, where to find OgmaNeo headers
#
if(OGMANEO_INCLUDE_DIR)
    # Already in cache, be silent
    set(OGMANEO_FIND_QUIETLY TRUE)
endif(OGMANEO_INCLUDE_DIR)

find_path(OGMANEO_INCLUDE_DIR ogmaneo/Hierarchy.h)

set(OGMANEO_NAMES ogmaneo OgmaNeo OGMANEO)
find_library(OGMANEO_LIBRARY NAMES ${OGMANEO_NAMES})

# Per-recommendation
set(OGMANEO_INCLUDE_DIRS "${OGMANEO_INCLUDE_DIR}")
set(OGMANEO_LIBRARIES    "${OGMANEO_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set OGMANEO_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OgmaNeo DEFAULT_MSG OGMANEO_LIBRARY OGMANEO_INCLUDE_DIR)