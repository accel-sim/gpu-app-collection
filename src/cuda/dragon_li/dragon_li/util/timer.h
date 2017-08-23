#pragma once

#include <sys/resource.h>
#include <sys/time.h>


namespace dragon_li {
namespace util {

struct CpuTimer
{
    struct timeval start;
    struct timeval stop;
    float secs;
    float usecs;

    CpuTimer() : secs(0.0), usecs(0.0) {}

    void Reset() {
        secs = 0.0;
        usecs = 0.0;
    }

    void Start()
    {
        gettimeofday(&start, NULL);
    }

    void Stop()
    {
        gettimeofday(&stop, NULL);
    }

    float ElapsedMillis()
    {
        float sec = stop.tv_sec - start.tv_sec;
        float usec = stop.tv_usec - start.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }
    void UpdateElapsedMillis() {
        
        float sec = stop.tv_sec - start.tv_sec;
        float usec = stop.tv_usec - start.tv_usec;

        secs += sec;
        usecs += usec;
    }
    float GetElapsedMillis() {
        
        return (secs * 1000) + (usecs / 1000);
    }

};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float totalMillis;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        totalMillis = 0.0;
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }

    void UpdateElapsedMillis() {
        totalMillis += ElapsedMillis();    
    }
    float GetElapsedMillis() {
        
        return totalMillis;
    }


};
}
}
