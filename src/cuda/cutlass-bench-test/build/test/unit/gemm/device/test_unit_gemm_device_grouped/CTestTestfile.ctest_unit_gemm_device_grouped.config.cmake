# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:cutlass_test_unit_gemm_device_grouped>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:cutlass_test_unit_gemm_device_grouped>)
endif()

add_test("ctest_unit_gemm_device_grouped" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" --gtest_output=xml:test_unit_gemm_device_grouped.gtest.xml)

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_unit_gemm_device_grouped" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_unit_gemm_device_grouped PROPERTIES DISABLED OFF)
