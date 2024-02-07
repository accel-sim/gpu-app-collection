# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /cutlass_test_unit_cute_msvc_compilation)
else()
  set(TEST_EXE_PATH cutlass_test_unit_cute_msvc_compilation)
endif()

add_test("ctest_unit_cute_msvc_compilation" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" --gtest_output=xml:test_unit_cute_msvc_compilation.gtest.xml)

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_unit_cute_msvc_compilation" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_unit_cute_msvc_compilation PROPERTIES DISABLED OFF)
