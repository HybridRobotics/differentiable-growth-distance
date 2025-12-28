add_subdirectory(internal_helpers)

# Mesh load test
add_executable(mesh_load_test
  "${CMAKE_CURRENT_SOURCE_DIR}/mesh_load_test.cc"
)
target_link_libraries(mesh_load_test PRIVATE ${DGD_LIB}_bench)
target_compile_features(mesh_load_test PUBLIC cxx_std_17)
target_compile_options(mesh_load_test
  PRIVATE
    -Wall -Wextra -Werror -Wconversion -Wpedantic
    -O3 -DNDEBUG
    -funroll-loops -march=native -flto
)

# Benchmark
add_executable(${DGD_LIB}_growth_distance_bm
  "${CMAKE_CURRENT_SOURCE_DIR}/growth_distance_bm.cc"
)
target_link_libraries(${DGD_LIB}_growth_distance_bm PRIVATE ${DGD_LIB}_bench)
target_compile_features(${DGD_LIB}_growth_distance_bm PUBLIC cxx_std_17)
target_compile_options(${DGD_LIB}_growth_distance_bm
  PRIVATE
    -Wall -Wextra -Werror -Wconversion -Wpedantic
    -O3 -DNDEBUG
    -funroll-loops -march=native -flto
)
