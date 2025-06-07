add_subdirectory(internal_helpers)

# Mesh load test
add_executable(mesh_load_test
  "${CMAKE_CURRENT_SOURCE_DIR}/mesh_load_test.cc"
)
target_link_libraries(mesh_load_test PRIVATE ${DGD_LIB}_internal_helper)

# Benchmark
add_executable(${DGD_LIB}_distance_bm
  "${CMAKE_CURRENT_SOURCE_DIR}/dgd_distance_bm.cc"
)
target_link_libraries(${DGD_LIB}_distance_bm PRIVATE ${DGD_LIB}_internal_helper)
