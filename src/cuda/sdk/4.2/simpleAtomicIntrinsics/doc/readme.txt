CUDA SDK Sample "simpleAtomics"

This code sample is meant to trivially exercise and demonstrate CUDA's global memory atomic functions:

atomicAdd()
atomicSub()
atomicExch()
atomicMax()
atomicMin()
atomicInc()
atomicdec()
atomicCAS()
atomicAnd()
atomicOr()
atomicXor()

This program requires compute capability 1.1.  To compile the code, therefore, note that the flag "-arch sm_11"
is passed to the nvcc compiler driver in the build step for simpleAtomics.cu.  To use atomics in your programs,
you must pass the same flag to the compiler.

Note that this program is meant to demonstrate the basics of using the atomic instructions, not to demonstrate 
a useful computation.