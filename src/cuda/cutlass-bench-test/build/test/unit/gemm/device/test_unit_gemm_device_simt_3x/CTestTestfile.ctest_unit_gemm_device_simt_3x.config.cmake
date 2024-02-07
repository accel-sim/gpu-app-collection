# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:cutlass_test_unit_gemm_device_simt_3x>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:cutlass_test_unit_gemm_device_simt_3x>)
endif()

add_test("ctest_unit_gemm_device_simt_3x" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" --gtest_output=xml:test_unit_gemm_device_simt_3x.gtest.xml)

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_unit_gemm_device_simt_3x" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_unit_gemm_device_simt_3x PROPERTIES DISABLED OFF)
