cmake_minimum_required(VERSION 3.15)

project(
  DifferentiableGrowthDistance
  VERSION 0.9.0
  DESCRIPTION "Differentiable growth distance algorithm for convex sets"
)

# Default to Release when no build type is specified.
# Explicit -DCMAKE_BUILD_TYPE=Debug always takes precedence.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified; defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()


message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
## User flags
#   Preprocessor definitions
option(DGD_USE_32BIT_FLOAT "Use 32-bit floating point" OFF)
option(DGD_VERBOSE_ITERATION "Enable verbose iteration-level printing" OFF)
option(DGD_EXTRACT_METRICS "Extract convergence metrics" OFF)

#   Installation rules
option(DGD_INSTALL_LIBS "Install library" ON)


## Packages
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Eigen3 5.0 REQUIRED)
# Prefer shared Qhull to avoid non-PIC static archive link failures for
# Python extension modules.
if(NOT DEFINED PCL_QHULL_REQUIRED_TYPE)
  set(PCL_QHULL_REQUIRED_TYPE "SHARED")
endif()
find_package(Qhull REQUIRED)


## DGD Core library
set(DGD_LIB dgd)
add_subdirectory(src)

if(DGD_INSTALL_LIBS AND CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT DGD_USE_32BIT_FLOAT)
  include(CMakePackageConfigHelpers)

  install(TARGETS ${DGD_LIB}
    EXPORT ${DGD_LIB}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  install(EXPORT ${DGD_LIB}Targets
    NAMESPACE ${DGD_LIB}::
    FILE "${DGD_LIB}Targets.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${DGD_LIB}
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dgdConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/${DGD_LIB}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${DGD_LIB}"
  )
  install(
    FILES "${PROJECT_BINARY_DIR}/${DGD_LIB}Config.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${DGD_LIB}"
  )

  # major.minor1.patch1 is compatible with major.minor2.patch2
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${DGD_LIB}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES "${PROJECT_BINARY_DIR}/${DGD_LIB}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${DGD_LIB}"
  )
  # Install FindQhull.cmake file
  install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindQhull.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${DGD_LIB}"
  )

  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/dgd"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
  )
endif()


# Tests, bechmarks, documentation, and Python bindings
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  option(DGD_BUILD_TESTS      "Build tests" ON)
  option(DGD_BUILD_BENCHMARKS "Build benchmarks (only for Release config with double precision)" OFF)
  option(DGD_BUILD_DOCS       "Build documentation" OFF)
  option(DGD_BUILD_PYTHON     "Build Python bindings (requires pybind11)" OFF)

  include(FetchContent)

  if(DGD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()

  if(DGD_BUILD_BENCHMARKS AND CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT DGD_USE_32BIT_FLOAT)
    add_subdirectory(benchmarks)
  endif()

  if(DGD_BUILD_DOCS)
    find_package(Doxygen)
    if(Doxygen_FOUND)
      message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE} -- ${DOXYGEN_VERSION}")
      add_subdirectory(docs EXCLUDE_FROM_ALL)
    else()
      message(STATUS "Doxygen not found; not building docs")
    endif()
  endif()

  if(DGD_BUILD_PYTHON)
    add_subdirectory(python)
  endif()

endif()
