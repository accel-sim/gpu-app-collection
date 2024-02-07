# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:18_ampere_fp64_tensorop_affine2_gemm>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:18_ampere_fp64_tensorop_affine2_gemm>)
endif()

add_test("ctest_examples_18_ampere_fp64_tensorop_affine2_gemm" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_18_ampere_fp64_tensorop_affine2_gemm" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_18_ampere_fp64_tensorop_affine2_gemm PROPERTIES DISABLED OFF)
