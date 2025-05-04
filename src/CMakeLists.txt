## DGD library
add_library(${DGD_LIB} STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/mesh_loader.cc"
)

add_library(${DGD_LIB}::${DGD_LIB} ALIAS ${DGD_LIB})

target_include_directories(${DGD_LIB}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../external"
  PUBLIC
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
    -Wall -Wextra -Werror -Wconversion -Wpedantic
    # $<$<CONFIG:Release>:-funroll-loops -march=native -mfpmath=sse -fopenmp>
    # $<$<AND:$<CONFIG:Release>,$<DGD_USE_FFAST_MATH>>:-ffast-math>
  PUBLIC
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)


# Preprocessor definitions
if (DGD_ENABLE_DIAGNOSTICS)
  target_compile_definitions(${DGD_LIB} PUBLIC DGD_PRINT_DIAGNOSTICS)
endif()

if (DGD_SINGLE_PRECISION)
  target_compile_definitions(${DGD_LIB} PUBLIC DGD_32BIT_FLOAT)
endif()
