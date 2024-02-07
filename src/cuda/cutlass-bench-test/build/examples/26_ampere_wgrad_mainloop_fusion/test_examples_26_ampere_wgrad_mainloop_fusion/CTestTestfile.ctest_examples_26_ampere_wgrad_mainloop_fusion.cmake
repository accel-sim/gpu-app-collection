# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /26_ampere_wgrad_mainloop_fusion)
else()
  set(TEST_EXE_PATH 26_ampere_wgrad_mainloop_fusion)
endif()

add_test("ctest_examples_26_ampere_wgrad_mainloop_fusion" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_26_ampere_wgrad_mainloop_fusion" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_26_ampere_wgrad_mainloop_fusion PROPERTIES DISABLED OFF)
