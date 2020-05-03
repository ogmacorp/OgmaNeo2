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
IF(OGMANEO_INCLUDE_DIR)
  # Already in cache, be silent
  SET(OGMANEO_FIND_QUIETLY TRUE)
ENDIF(OGMANEO_INCLUDE_DIR)

FIND_PATH(OGMANEO_INCLUDE_DIR ogmaneo/Hierarchy.h)

SET(OGMANEO_NAMES ogmaneo OgmaNeo OGMANEO)
FIND_LIBRARY(OGMANEO_LIBRARY NAMES ${OGMANEO_NAMES})

# Per-recommendation
SET(OGMANEO_INCLUDE_DIRS "${OGMANEO_INCLUDE_DIR}")
SET(OGMANEO_LIBRARIES    "${OGMANEO_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set OGMANEO_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OgmaNeo DEFAULT_MSG OGMANEO_LIBRARY OGMANEO_INCLUDE_DIR)