# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:50_hopper_gemm_with_epilogue_swizzle>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:50_hopper_gemm_with_epilogue_swizzle>)
endif()

add_test("ctest_examples_50_hopper_gemm_with_epilogue_swizzle" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_50_hopper_gemm_with_epilogue_swizzle" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_50_hopper_gemm_with_epilogue_swizzle PROPERTIES DISABLED OFF)
