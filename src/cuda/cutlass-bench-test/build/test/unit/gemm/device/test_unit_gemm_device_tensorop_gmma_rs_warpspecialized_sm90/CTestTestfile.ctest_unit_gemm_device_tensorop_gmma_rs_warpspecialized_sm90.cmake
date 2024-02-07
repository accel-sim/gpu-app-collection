# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /cutlass_test_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90)
else()
  set(TEST_EXE_PATH cutlass_test_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90)
endif()

add_test("ctest_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" --gtest_output=xml:test_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90.gtest.xml)

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_unit_gemm_device_tensorop_gmma_rs_warpspecialized_sm90 PROPERTIES DISABLED OFF)
