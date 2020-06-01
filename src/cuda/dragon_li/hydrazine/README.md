#Overview
A library of common operations needed for C++ and CUDA development

#Libraries
##C++
##Threading
A wrapper around pthreads providing a message passing interface rather than a locking based interface.

##Argument Parser
A parser for command line arguments.

##B-Tree
A replacement for std::map implementing the complete ISO/IEC 14882:2003 standard with a Btree relying on mmapped pages.

##Debugging
Conditional debugging messages as well as a more informative version of assert (assert.h).

##Timer
Interface to high precision linux timers as well as rdtsc timers on x86 processors.

##Active Timer
A wrapper around pthreads providing an asynchronous split-phase interface rather than a locking interface.

##Serialization
An interface for serializing classes to contiguous arrays and unpacking them.

##XML Parser
Basic parser for XML.

#CUDA
##Error Handling
Wrappers to convert CUDA error codes to exceptions.

##Vector
Interface for std::vector where the vector object is manipulated on the host, but the memory actually resides on the CUDA device. The idea is to get away from using malloc/free/memcpy for allocation and data transfers. Provides host-dereferencable iterators.

#Contact
##Authors
Gregory Diamos [email](mailto:solusstultus@gmail.com)
