#   DGD library
add_library(${LIB_CORE}
  "${PROJECT_SOURCE_DIR}/src/mesh_loader.cc"
)

target_include_directories(${LIB_CORE}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include/"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/tinyobjloader/"
)

target_link_libraries(${LIB_CORE}
  PRIVATE
    Eigen3::Eigen
    QHULL::QHULL
)
