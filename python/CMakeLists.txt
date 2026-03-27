## Python bindings for the DGD library using pybind11

# Use the modern CMake Python finder so that -DPYTHON_EXECUTABLE is respected
# and the correct interpreter/libs are always used together.
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(pybind11)

# OPT_SIZE is appropriate for wrapper code because the hot paths live in libdgd.
# THIN_LTO keeps link-time optimisation available without adding custom flags.
pybind11_add_module(_dgd_core OPT_SIZE THIN_LTO
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/types.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/settings.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/output.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry_concrete.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cc"
)

target_link_libraries(_dgd_core PRIVATE ${DGD_LIB})

target_compile_features(_dgd_core PRIVATE cxx_std_17)
set_target_properties(_dgd_core PROPERTIES
  CXX_EXTENSIONS OFF
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dgd"
)

install(
  TARGETS _dgd_core
  LIBRARY DESTINATION dgd
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_dgd_core
    PRIVATE
      -Wall
      -Wextra
      -Wconversion
      -Wpedantic
      -Wnon-virtual-dtor
  )
endif()
# pybind11_add_module already handles config-specific optimization flags.
