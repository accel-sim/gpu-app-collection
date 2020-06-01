// need special implementation for WIN

#include <time.h>
#include <sys/time.h>

class Timer
{
  struct timespec m_start;
  struct timespec m_stop;
public: 
  Timer() 
  {
    clock_gettime(CLOCK_MONOTONIC, &m_start);
  }

  void stop()
  {
    clock_gettime(CLOCK_MONOTONIC, &m_stop);
  }

  double elapsed_sec()
  {
    double dt = (m_stop.tv_sec - m_start.tv_sec) + (double) (m_stop.tv_nsec - m_start.tv_nsec) * 1e-9;
    return dt;
  }
};
