# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:20_simt_canonical>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:20_simt_canonical>)
endif()

add_test("ctest_examples_20_simt_canonical" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_20_simt_canonical" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_20_simt_canonical PROPERTIES DISABLED OFF)
