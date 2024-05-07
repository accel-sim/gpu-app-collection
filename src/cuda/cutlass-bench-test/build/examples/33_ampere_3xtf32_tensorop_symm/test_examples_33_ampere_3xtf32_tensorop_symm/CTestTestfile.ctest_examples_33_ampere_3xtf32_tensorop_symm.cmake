# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /33_ampere_3xtf32_tensorop_symm)
else()
  set(TEST_EXE_PATH 33_ampere_3xtf32_tensorop_symm)
endif()

add_test("ctest_examples_33_ampere_3xtf32_tensorop_symm" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_33_ampere_3xtf32_tensorop_symm" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_33_ampere_3xtf32_tensorop_symm PROPERTIES DISABLED OFF)
