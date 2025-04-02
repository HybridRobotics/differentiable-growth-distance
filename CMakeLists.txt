## CMake version
cmake_minimum_required(VERSION 3.15)


## User flags
#   Build flags
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS
  "Build benchmarks (only built if BUILD_TESTS is enabled)"
  ON
)
option(BUILD_APPS "Build examples" OFF)
option(BUILD_DOCS "Add documentation" OFF)
#   Compile flags
option(USE_FFAST_MATH "Add -ffast-math compile option" OFF)
#   Preprocessor definitions
option(ENABLE_DIAGNOSTICS "Print solution diagnostics" OFF)


## Project name and version
project(
  DifferentiableGrowthDistance
  VERSION 0.1.0
  DESCRIPTION "Differentiable growth distance algorithm for convex sets"
  LANGUAGES CXX
)


## Compiler flags
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} \
  -pedantic"
)
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} \
  -O0 -Wall -Wextra -Wconversion -g"
)
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} \
  -O3 -Werror -DNDEBUG -funroll-loops -march=native -mfpmath=sse -fopenmp"
)
# Note: -ffast-math (-fno-math-errno, -Ofast) breaks IEEE compliance
#       and enables -ffinite-math-only.
if (USE_FFAST_MATH)
  set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} \
    -ffast-math"
  )
endif()


## Packages
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(FetchContent)

find_package(Eigen3 3.4.90 REQUIRED NO_MODULE)
find_package(Qhull 2020.2 REQUIRED)


## DGD subdirectories
set(LIB_CORE dgd)
#   Preprocessor definitions
if (NOT ENABLE_DIAGNOSTICS)
  add_compile_definitions(DONOT_PRINT_DIAGNOSTICS)
endif()

add_subdirectory(src)


## Tests
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  if (BUILD_BENCHMARKS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_subdirectory(benchmarks)
  endif()
endif()


## Apps
if (BUILD_APPS)
  add_subdirectory(apps)
endif()


## Documentation
if (BUILD_DOCS)
  find_package(Doxygen)
  if (Doxygen_FOUND)
    message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE} -- ${DOXYGEN_VERSION}")
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found; not building docs")
  endif()
endif()
