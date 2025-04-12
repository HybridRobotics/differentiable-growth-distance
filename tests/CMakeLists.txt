# Testing library (GoogleTest v1.16.0)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(${LIB_CORE}_test
  "${PROJECT_SOURCE_DIR}/tests/geometry_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/utils_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/graham_scan_test.cc"
  "${PROJECT_SOURCE_DIR}/tests/mesh_loader_test.cc"
)

target_link_libraries(${LIB_CORE}_test
  PRIVATE
    ${LIB_CORE}
    GTest::gtest_main
)

target_compile_options(${LIB_CORE}_test
  PRIVATE
    -pedantic
    $<$<CONFIG:Debug>:-Wall -Wextra>
    $<$<CONFIG:Release>:-Werror>
)


include(GoogleTest)
gtest_discover_tests(${LIB_CORE}_test)
