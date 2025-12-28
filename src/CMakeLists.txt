## DGD library
add_library(${DGD_LIB} STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/graham_scan.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/mesh_loader.cc"
)

add_library(${DGD_LIB}::${DGD_LIB} ALIAS ${DGD_LIB})

target_include_directories(${DGD_LIB}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../external"
  PUBLIC
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${DGD_LIB}
  PRIVATE
    QHULL::QHULL
  PUBLIC
    Eigen3::Eigen
)


# Compiler flags
# Use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON to check compile options
target_compile_features(${DGD_LIB} PUBLIC cxx_std_17)
set_target_properties(${DGD_LIB} PROPERTIES CXX_EXTENSIONS OFF)

# Note: -ffast-math (-fno-math-errno, -Ofast) breaks IEEE compliance
#       and enables -ffinite-math-only.
target_compile_options(${DGD_LIB}
  PRIVATE
    -Wall -Wextra -Werror -Wconversion -Wpedantic -fdiagnostics-color=always
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Release>:-funroll-loops -march=native -flto>
)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set_target_properties(${DGD_LIB} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


# Preprocessor definitions
if (DGD_USE_32BIT_FLOAT)
  target_compile_definitions(${DGD_LIB} PUBLIC DGD_USE_32BIT_FLOAT)
endif()
