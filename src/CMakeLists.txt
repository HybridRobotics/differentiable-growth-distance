## DGD library
add_library(${LIB_CORE}
  "${PROJECT_SOURCE_DIR}/src/mesh_loader.cc"
)

target_include_directories(${LIB_CORE}
  PRIVATE
    "${PROJECT_SOURCE_DIR}/tinyobjloader/"
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include/"
)

target_link_libraries(${LIB_CORE}
  PRIVATE
    QHULL::QHULL
  PUBLIC
    Eigen3::Eigen
)

## Compiler flags
target_compile_features(${LIB_CORE} PUBLIC cxx_std_14)
set_target_properties(${LIB_CORE} PROPERTIES CXX_EXTENSIONS OFF)

# Note: -ffast-math (-fno-math-errno, -Ofast) breaks IEEE compliance
#       and enables -ffinite-math-only.
target_compile_options(${LIB_CORE}
  PRIVATE
    -pedantic
    $<$<CONFIG:Debug>:-Wall -Wextra>
    $<$<CONFIG:Release>:-Werror>
    # $<$<CONFIG:Release>:-funroll-loops -march=native -mfpmath=sse -fopenmp>
    # $<$<AND:$<CONFIG:Release>,$<USE_FFAST_MATH>>:-ffast-math>
  PUBLIC
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

# Preprocessor definitions
if (ENABLE_DIAGNOSTICS)
  target_compile_definitions(${LIB_CORE} PUBLIC DGD_PRINT_DIAGNOSTICS)
endif()

if (SINGLE_PRECISION)
  target_compile_definitions(${LIB_CORE} PUBLIC DGD_32_BITS)
endif()

# Use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON to check compile options
