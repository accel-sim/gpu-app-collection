
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>


/*----------- using cycle counter ------------*/
     __inline__ uint64_t rdtsc() 
     {
#ifdef __x86_64__
          uint32_t lo, hi;
             /* We cannot use "=A", since this would use %rax on x86_64 */
             __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
                return (uint64_t)hi << 32 | lo;
#endif
#ifdef __riscv
          uint64_t cycle;
          asm volatile (
            "rdcycle %0"
            : "=r" (cycle)
          );
          return cycle;
#endif
     }

unsigned long long start_cycles;
#define startCycle() (start_cycles = rdtsc())
#define stopCycle(cycles) (cycles = rdtsc()-start_cycles)

/*--------- using gettimeofday ------------*/

#include <sys/time.h>

struct timeval starttime;
struct timeval endtime;

#define startTime() \
{ \
  gettimeofday(&starttime, 0); \
}
#define stopTime(valusecs) \
{ \
  gettimeofday(&endtime, 0); \
  valusecs = (endtime.tv_sec-starttime.tv_sec)*1000000+endtime.tv_usec-starttime.tv_usec; \
}
