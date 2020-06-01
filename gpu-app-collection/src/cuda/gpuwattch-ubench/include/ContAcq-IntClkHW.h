
#include <stdio.h>
#include <NIDAQmx.h>
#include <stdlib.h>

TaskHandle LaunchDAQ();
void TurnOffDAQ(TaskHandle taskHandle, float endtime);
