==============================================================================
                   __   __ ___________                 _                        
                   \ \ / //  ___| ___ \               | |                       
                    \ V / \ `--.| |_/ / ___ _ __   ___| |__                     
                    /   \  `--. \ ___ \/ _ \ '_ \ / __| '_ \                    
                   / /^\ \/\__/ / |_/ /  __/ | | | (__| | | |                   
                   \/   \/\____/\____/ \___|_| |_|\___|_| |_|                   

==============================================================================
Background
==============================================================================

XSBench is a simple application that executes only the most
computationally expensive steps of Monte Carlo particle transport,
the calculation of macroscopic cross sections, in an effort to
expose bottlenecks within multi-core, shared memory architectures.

In a particle transport simulation, every time a particle changes
energy or crosses a material boundary, a new macroscopic cross
section must be calculated. The time spent looking up and calculating
the required cross section information often accounts for well over
half of the total active runtime of the simulation. XSBench uses a
unionized energy grid to facilitate cross section lookups for
randomized particle energies. There are a similar number of energy
grid points, material types, and nuclides as are used in the
Hoogenboom-Martin benchmark. The probability of particles residing
in any given material is weighted based on that material's commonality
inside the Hoogenboom-Martin benchmark geometry.

The end result is that the essential computational conditions and
tasks of fully featured Monte Carlo transport codes are retained
in XSBench, without the additional complexity and overhead inherent
in a fully featured code. This provides a much simpler and clearer
platform for stressing different architectures, and ultimately for
making determinations as to where hardware bottlenecks occur as
cores are added.

==============================================================================
Quick Start
==============================================================================

To compile and run XSBench with default settings, use the following
commands:

>$ make

>$ ./XSBench

By default, XSBench will with 1 thread per hardware core. If the
architecture supports hyperthreading, two threads will be run per
core.

To manually alter the number of threads used, run the code with the
desired number of threads as an argument:

>$ ./XSBench 4

To alter the Hoogenboom-Martin specification used (small or large),
use the second argument to specify either "Small" or "Large". This
corresponds to the number of nuclides present in the fuel region.
The small version has 34 fuel nuclides, whereas the large version
has 321 fuel nuclides. This significantly slows down the runtime
of the program as the data structures are much larger, and more
lookups are required whenever a lookup occurs in a fuel material.
Note that the program defaults to "Large" if no specification is
made. Example:

>$ ./XSBench 4 Small


==============================================================================
Debugging, Optimization & Profiling
==============================================================================

There are also a number of switches that can be set in the makefile.

Here is a sample of the control panel at the top of the makefile:

COMPILER = gnu
OPTIMIZE = yes
DEBUG    = no
PROFILE  = no
PAPI     = no

Optimization enables the -O3 optimzation flag.

Debugging enables the -g flag.

Profling enables the -pg flag.

The PAPI flag is explained below.


==============================================================================
PAPI Performance Counters
==============================================================================

By default, PAPI is disabled.

To enable PAPI, open the XSBench_header.h file and add (or uncomment)
the following definition to the file:

"#define __PAPI"

Then, enable papi in the makefile:

PAPI     = yes

Note that you may need to change the relevant library paths for papi
to work (as these are dependent on your machine).
The library path can be specified in the makefile, and the
header path is specified in the XSBench_header.h file.



==============================================================================
Running on ANL BlueGene/Q (Vesta & Mira)
==============================================================================

Compilation is done using the included makefile, as follows:

>$ make MACHINE=bluegene


Note that the INFO macro in the XSbench_header.h file should be
set to 0 when running on BG/Q to remove the run status portions of
the output, which cuts down on unnecessary file I/O, i.e.:

#define INFO 0


Also, note that you may need to add the following line to
your .soft file in order to use the mpicc compiler wrapper:

+mpiwrapper-gcc

Then, be sure to use the "resoft" command to update your software. 

When running in c16 mode, the maximum number of gridpoints per nuclide
is 900 (when running in "Large" mode). More points will cause the 1GB 
memory limit to be broken.


A basic test run on 1 node can be achieved (assuming you have an allocation)
using the makefile and the following command:

>$ make bgqrun

Further information on queuing can be found at:

https://www.alcf.anl.gov/resource-guides/vesta-queuing



==============================================================================
Adding Extra Flops
==============================================================================

One of the areas we're investigating is the effect of adding additional
flops per each load from the nuclide xs arrays. Adding flops has so far
shown to increase scaling, indicating that there is in fact a bottleneck
being caused by the memory loads.

To enable this feature, go to the XSBench_header.h file and uncomment
out the "#define ADD_EXTRAS" line.

By default, there are no "extra" flops in the benchmark. To add these in,
open the XSbench_header.h file and change the EXTRA_FLOPS definition to
the desired number of additional flops. i.e.,

"#define EXTRA_FLOPS 10"

Note that even just 5-10 extra flops will greatly increase the running
time of the program.



==============================================================================
Adding Extra Memory Loads
==============================================================================

One of the areas we're investigating is the effect of adding
additional memory loads per each required load from the nuclide xs
arrays. Adding loads has so far been rather inconclusive, as randomizing
the loads requires a number of flops be performed to generate the random
index number to load from. 

To enable this feature, go to the XSBench_header.h file and uncomment
out the "#define ADD_EXTRAS" line.

By default, there are no "extra" loads in the benchmark. To add these in,
open the XSbench_header.h file and change the EXTRA_LOADS definition to
the desired number of additional loads. i.e.,

"#define EXTRA_LOADS 10"

Note that even just a few extra loads will greatly increase the running
time of the program.
