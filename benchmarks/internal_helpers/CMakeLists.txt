## Internal benchmark helper library
add_library(${DGD_LIB}_bench STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/set_generator.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/mesh_loader.cc"
)

target_include_directories(${DGD_LIB}_bench
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../"
)

target_link_libraries(${DGD_LIB}_bench
  PUBLIC
    Eigen3::Eigen
    ${DGD_LIB}::${DGD_LIB}
)

# Compiler flags
target_compile_features(${DGD_LIB}_bench PUBLIC cxx_std_17)
set_target_properties(${DGD_LIB}_bench PROPERTIES CXX_EXTENSIONS OFF)

# Platform-specific compiler flags for warnings and optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(${DGD_LIB}_bench
    PRIVATE
      -Wall -Wextra -Werror -Wconversion -Wpedantic
      -O3 -DNDEBUG
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(${DGD_LIB}_bench
    PRIVATE
      /W4 /WX /O2 /DNDEBUG /permissive-
  )
endif()
