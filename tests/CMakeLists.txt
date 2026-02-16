# Testing library (GoogleTest v1.16.0)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
# Prevent Gtest from being added to the ALL target (installation)
set(INSTALL_GTEST OFF)
set(gtest_force_shared_libs OFF CACHE INTERNAL "")
set(gtest_build_tests OFF CACHE INTERNAL "")
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)


add_executable(${DGD_LIB}_test
  "${PROJECT_SOURCE_DIR}/tests/gradient_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/graham_scan_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/growth_distance_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/kkt_nullspace_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/local_geometry_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/mesh_loader_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/support_function_test.cc"
)

target_link_libraries(${DGD_LIB}_test
  PRIVATE
    ${DGD_LIB}
    GTest::gtest_main
)

target_compile_options(${DGD_LIB}_test
  PRIVATE
    -Wall -Wextra -Werror -Wconversion -Wpedantic -Wnon-virtual-dtor -Wsuggest-override
)


include(GoogleTest)
gtest_discover_tests(${DGD_LIB}_test)
