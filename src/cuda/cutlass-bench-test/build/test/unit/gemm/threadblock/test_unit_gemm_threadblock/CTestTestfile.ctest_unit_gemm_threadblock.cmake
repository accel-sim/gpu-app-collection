# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /cutlass_test_unit_gemm_threadblock)
else()
  set(TEST_EXE_PATH cutlass_test_unit_gemm_threadblock)
endif()

add_test("ctest_unit_gemm_threadblock" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" --gtest_output=xml:test_unit_gemm_threadblock.gtest.xml)

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_unit_gemm_threadblock" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_unit_gemm_threadblock PROPERTIES DISABLED OFF)
