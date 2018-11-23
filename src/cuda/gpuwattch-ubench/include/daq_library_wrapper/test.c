#include "ContAcq-IntClk.h"

void main()
{
	TaskHandle ta = LaunchDAQ();
	sleep(10);
	TurnOffDAQ(ta);
}

