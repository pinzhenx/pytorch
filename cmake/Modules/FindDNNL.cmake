# - Try to find DNNL
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  DNNL_FOUND          : set to true if dnnl is found.
#  DNNL_INCLUDE_DIR    : path to dnnl include dir.
#  DNNL_LIBRARIES      : list of libraries for dnnl
#
# The following variables are used:
#  DNNL_USE_NATIVE_ARCH : Whether native CPU instructions should be used in DNNL. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT DNNL_FOUND)

SET(DNNL_LIBRARIES)
SET(DNNL_INCLUDE_DIR)

IF(MSVC)
  MESSAGE(STATUS "DNNL needs omp 3+ which is not supported in MSVC so far")
  RETURN()
ENDIF(MSVC)

SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
SET(DNNL_ROOT "${IDEEP_ROOT}/mkl-dnn")

FIND_PACKAGE(BLAS)
FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
FIND_PATH(DNNL_INCLUDE_DIR dnnl.hpp dnnl.h PATHS ${DNNL_ROOT} PATH_SUFFIXES include)
IF (NOT DNNL_INCLUDE_DIR)
  EXECUTE_PROCESS(COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init dnnl WORKING_DIRECTORY ${IDEEP_ROOT})
  FIND_PATH(DNNL_INCLUDE_DIR dnnl.hpp dnnl.h PATHS ${DNNL_ROOT} PATH_SUFFIXES include)
ENDIF(NOT DNNL_INCLUDE_DIR)

IF (NOT IDEEP_INCLUDE_DIR OR NOT DNNL_INCLUDE_DIR)
  MESSAGE(STATUS "DNNL source files not found!")
  RETURN()
ENDIF(NOT IDEEP_INCLUDE_DIR OR NOT DNNL_INCLUDE_DIR)
LIST(APPEND DNNL_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})
# XPZ: TODO: remove MKL dependency
IF(MKL_FOUND)
  ADD_DEFINITIONS(-DIDEEP_USE_MKL)
  # Append to dnnl dependencies
  LIST(APPEND DNNL_LIBRARIES ${MKL_LIBRARIES})
  LIST(APPEND DNNL_INCLUDE_DIR ${MKL_INCLUDE_DIR})
ELSE(MKL_FOUND)
  SET(DNNL_USE_MKL "NONE" CACHE STRING "" FORCE)
ENDIF(MKL_FOUND)

SET(MKL_cmake_included TRUE)
IF (NOT DNNL_CPU_RUNTIME)
  SET(DNNL_CPU_RUNTIME "OMP" CACHE STRING "")
ELSEIF (DNNL_CPU_RUNTIME STREQUAL "TBB")
  IF (USE_TBB)
    MESSAGE(STATUS "DNNL is using TBB")

    SET(TBB_cmake_included TRUE)
    SET(Threading_cmake_included TRUE)

    REMOVE_DEFINITIONS(-DDNNL_THR)
    ADD_DEFINITIONS(-DDNNL_THR=DNNL_THR_TBB)

    SET(TBB_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/tbb/include")
    INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})
    LIST(APPEND EXTRA_SHARED_LIBS tbb)
  ELSE()
    MESSAGE(FATAL_ERROR "DNNL_CPU_RUNTIME is set to TBB but TBB is not used")
  ENDIF()
ENDIF()
MESSAGE(STATUS "DNNL_CPU_RUNTIME = ${DNNL_CPU_RUNTIME}")

SET(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
SET(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
SET(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
SET(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
IF(DNNL_USE_NATIVE_ARCH)  # Disable HostOpts in DNNL unless DNNL_USE_NATIVE_ARCH is set.
  SET(ARCH_OPT_FLAGS "HostOpts" CACHE STRING "" FORCE)
ELSE()
  IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(ARCH_OPT_FLAGS "-msse4" CACHE STRING "" FORCE)
  ELSE()
    SET(ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
  ENDIF()
ENDIF()

ADD_SUBDIRECTORY(${DNNL_ROOT})
IF(NOT TARGET dnnl)
  MESSAGE("Failed to include DNNL target")
  RETURN()
ENDIF(NOT TARGET dnnl)
IF(MKL_FOUND)
  SET(USE_MKL_CBLAS -DUSE_MKL)
  IF(USE_DNNL_CBLAS)
    LIST(APPEND USE_MKL_CBLAS -DUSE_CBLAS)
  ENDIF(USE_DNNL_CBLAS)
  TARGET_COMPILE_DEFINITIONS(dnnl PRIVATE USE_MKL_CBLAS)
ENDIF(MKL_FOUND)
IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-maybe-uninitialized)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-strict-overflow)
  TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-error=strict-overflow)
ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-tautological-compare)
LIST(APPEND DNNL_LIBRARIES dnnl)

SET(DNNL_FOUND TRUE)
MESSAGE(STATUS "Found DNNL: TRUE")

ENDIF(NOT DNNL_FOUND)
