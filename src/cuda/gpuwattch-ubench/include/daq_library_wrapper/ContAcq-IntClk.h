
#include <stdio.h>
#include <NIDAQmx.h>
#include <stdlib.h>

#define USE_PROFILING

#define START_SIGN 0
#define SIGN_START 0
#define END_SIGN 20
#define SIGN_END 20
TaskHandle LaunchDAQ();
void TurnOffDAQ(TaskHandle taskHandle, float endtime);
void TurnOffDAQ(TaskHandle taskHandle);
void InsertSign(TaskHandle taskHandle, float signature);
