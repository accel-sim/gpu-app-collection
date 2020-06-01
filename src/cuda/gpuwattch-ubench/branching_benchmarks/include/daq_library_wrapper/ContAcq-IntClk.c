/*********************************************************************
*
* ANSI C Example program:
*    ContAcq-IntClk.c
*
* Example Category:
*    AI
*
* Description:
*    This example demonstrates how to acquire a continuous amount of
*    data using the DAQ device's internal clock.
*
* Instructions for Running:
*    1. Select the physical channel to correspond to where your
*       signal is input on the DAQ device.
*    2. Enter the minimum and maximum voltage range.
*    Note: For better accuracy try to match the input range to the
*          expected voltage level of the measured signal.
*    3. Set the rate of the acquisition. Also set the Samples per
*       Channel control. This will determine how many samples are
*       read each time the while loop iterates. This also determines
*       how many points are plotted on the graph each iteration.
*    Note: The rate should be at least twice as fast as the maximum
*          frequency component of the signal being acquired.
*
* Steps:
*    1. Create a task.
*    2. Create an analog input voltage channel.
*    3. Set the rate for the sample clock. Additionally, define the
*       sample mode to be continuous.
*    4. Call the Start function to start the acquistion.
*    5. Read the data in a loop until the stop button is pressed or
*       an error occurs.
*    6. Call the Clear Task function to clear the task.
*    7. Display an error if any.
*
* I/O Connections Overview:
*    Make sure your signal input terminal matches the Physical
*    Channel I/O control. For further connection information, refer
*    to your hardware reference manual.
*
*********************************************************************/

#include "ContAcq-IntClk.h"
#include<string.h>
#include<sys/time.h>

#define DAQmxErrChk(functionCall) if( DAQmxFailed(error=(functionCall)) ) goto Error; else
float senseResistorATX = 0.025; //25 mOhms
int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);
int32 CVICALLBACK DoneCallback(TaskHandle taskHandle, int32 status, void *callbackData);
struct timeval *start, *end;
#define BUFFER_SIZE 70000000*2
#define NUM_CHN 5

FILE * fp;
bool file_open = false;
float64   *data;
unsigned long head;
TaskHandle LaunchDAQ() {

		int32       error=0;
		TaskHandle  taskHandle=0;
		char        errBuff[2048]={'\0'};
		/**********************************************/
		// DAQmx Configure Code
		/*********************************************/
		DAQmxErrChk (DAQmxCreateTask("",&taskHandle));
		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai0","",DAQmx_Val_Cfg_Default, 0, 7,    DAQmx_Val_Volts,NULL));	//ATX 12 V Supply
		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai1","",DAQmx_Val_Cfg_Default, 0, 1.25, DAQmx_Val_Volts,NULL));	//ATX 12 V Sense Resistor Drop
		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai2","",DAQmx_Val_Cfg_Default, 0, 7,    DAQmx_Val_Volts,NULL));	//12 V PCIE Supply
		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai3","",DAQmx_Val_Cfg_Default, 0, 1.25, DAQmx_Val_Volts,NULL));	//12 V PCIE Sense Resistor Drop
		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai4","",DAQmx_Val_Cfg_Default, 0, 1.25,    DAQmx_Val_Volts,NULL));	//Another 12 ATX current sense resistor
		//DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai5","",DAQmx_Val_Cfg_Default, 0, 1.25, DAQmx_Val_Volts,NULL));	//3.3 V Aux PCIE Sense Resistor Drop
//		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai6","",DAQmx_Val_Cfg_Default, 0, 4,    DAQmx_Val_Volts,NULL));	//3.3 V PCIE Supply
//		DAQmxErrChk (DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai7","",DAQmx_Val_Cfg_Default, 0, 1.25, DAQmx_Val_Volts,NULL));	//3.3 V PCIE Sense Resistor Drop 
		////DAQmxErrChk (DAQmxCfgSampClkTiming(taskHandle,"",500000.0,DAQmx_Val_Rising,DAQmx_Val_ContSamps,10000)); //sampsPerChanToAcquire
		//DAQmxErrChk (DAQmxCfgSampClkTiming(taskHandle,"",500000.0,DAQmx_Val_Rising,DAQmx_Val_ContSamps,10000)); //sampsPerChanToAcquire
		DAQmxErrChk (DAQmxCfgSampClkTiming(taskHandle,"",2400000,DAQmx_Val_Rising,DAQmx_Val_ContSamps,3000000)); //sampsPerChanToAcquire

		DAQmxErrChk (DAQmxRegisterEveryNSamplesEvent(taskHandle,DAQmx_Val_Acquired_Into_Buffer,100000,0,EveryNCallback,NULL));
		//DAQmxErrChk (DAQmxRegisterEveryNSamplesEvent(taskHandle,DAQmx_Val_Acquired_Into_Buffer,10000,0,EveryNCallback,NULL));
		DAQmxErrChk (DAQmxRegisterDoneEvent(taskHandle,0,DoneCallback,NULL));


		//printf("Acquiring samples continuously. Press Enter to interrupt\n");
		if (file_open)
			fp = fopen("powerdatafile","a");
		else {
			fp = fopen("powerdatafile","w");
			file_open = true;
		}
		//printf("fp = %d", fp);
		data = (float64 *)malloc(BUFFER_SIZE * sizeof(float64));
		head = 0;
		if (data == NULL) {
			//printf("malloc failed!\n");
			exit(4);
		}
		memset(data, 0, BUFFER_SIZE * sizeof(float64));

		/*********************************************/
		// DAQmx Start Code
		/*********************************************/
		DAQmxErrChk (DAQmxStartTask(taskHandle));
		return taskHandle;

		Error:
		if( DAQmxFailed(error) )
			DAQmxGetExtendedErrorInfo(errBuff,2047);
		if( taskHandle!=0 ) {
			/*********************************************/
			// DAQmx Stop Code
			/*********************************************/
			DAQmxStopTask(taskHandle);
			DAQmxClearTask(taskHandle);
		}
		if( DAQmxFailed(error) )
			//printf("DAQmx Error: %s\n",errBuff);
		//printf("End of program, press Enter key to quit\n");
		return 0;

}

int  totalRead=0;

void TurnOffDAQ(TaskHandle taskHandle, float endtime){
		int32       error=0;
		int32       read=0;
		char        errBuff[2048]={'\0'};
		int my_error = 1;

	DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,-1,10.0,DAQmx_Val_GroupByScanNumber,&data[head],BUFFER_SIZE-head,&read,NULL));
	my_error = 0;
	DAQmxStopTask(taskHandle);
	DAQmxClearTask(taskHandle);
	if( read>0 ) {

//		printf("Acquired %d samples from TurnOffDAQ. Total %d\r",read,totalRead+=read);
		head += (read - (read%NUM_CHN));
		if (head > BUFFER_SIZE) {
			//printf ("buffer overflow!\n");
			exit(5);
		}
		fflush(stdout);

	}

	//fflush(stdout);
	unsigned long i;
	for (i = 0; i < head - NUM_CHN; i+=NUM_CHN) {
		if(fprintf(fp, "%f,%f\n",2*data[i+1]*((data[i]+data[i+4])/senseResistorATX),2*data[i+3]*(data[i+2]/senseResistorATX)) < 0){     
		printf ("error\n");

		}
	} 
	fprintf(fp,"%f\n", endtime);

Error:
	if(my_error){
		DAQmxStopTask(taskHandle);
		DAQmxClearTask(taskHandle);
     
	}
	//printf("my_error = %d\n", my_error);
	if( DAQmxFailed(error) ) {
		DAQmxGetExtendedErrorInfo(errBuff,2048);
		
		// DAQmx Stop Code
	
		printf("DAQmx Error: %s\n",errBuff);
	}
	fclose(fp);
	free(data);
	return;
}


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData)
{
	int32       error=0;
	char        errBuff[2048]={'\0'};
	int32       read=0;

	/*********************************************/
	// DAQmx Read Code
	/*********************************************/
	static int count = 0;
//	printf ("Call Back %d\n", count);
	count ++;
	//printf("Start of EveryNCallback\n");
	//DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,1000000,10.0,DAQmx_Val_GroupByScanNumber,data,12000000,&read,NULL));
	DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,-1,10.0,DAQmx_Val_GroupByScanNumber,&data[head],BUFFER_SIZE-head,&read,NULL));
	if( read>0 ) {

		head += (read - (read%NUM_CHN));
		if (head > BUFFER_SIZE) {
			exit(5);
		}
	}

//	printf("End of EveryNCallback\n");
Error:
//	printf("EveryNCallback: Inside Error\n");
	if( DAQmxFailed(error) ) {
		DAQmxGetExtendedErrorInfo(errBuff,2048);
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskHandle);
		DAQmxClearTask(taskHandle);
		printf("DAQmx Error: %s\n",errBuff);
	}
	return 0;
}

int32 CVICALLBACK DoneCallback(TaskHandle taskHandle, int32 status, void *callbackData)
{
	int32   error=0;
	char    errBuff[2048]={'\0'};
	int32       read=0;

	DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,-1,10.0,DAQmx_Val_GroupByScanNumber,data,6400000,&read,NULL));
	printf ("DoneCallback\n");
	if( read>0 ) {

		printf("Acquired %d samples. Total %d\r",read,totalRead+=read);
		fflush(stdout);
		int i;
		for (i = 0; i < read-8; i+=8) {
			if(fprintf(fp, "DoneCallback%f,%f,%f\t %f,%f,%f\t %f,%f,%f\t %f,%f,%f\n",
			data[i], data[i+1], 2*data[i+1]*(data[i]/senseResistorATX), 
			data[i+2], data[i+3], 2*data[i+3]*(data[i+2]/senseResistorATX), 
			data[i+4], data[i+5], data[i+5]*(data[i+4]/senseResistorATX),
			data[i+6], data[i+7], data[i+7]*(data[i+6]/senseResistorATX)) < 0 ) {

			printf ("error\n");

		}
		/*	fprintf(fp, "%f,%f\t %f,%f\t %f,%f\t %f,%f\n",
			data[i], data[i+1], 
			data[i+2], data[i+3], 
			data[i+4], data[i+5], 
			data[i+6], data[i+7]); */
		}
	}
		fflush(fp);
	// Check to see if an error stopped the task.
	DAQmxErrChk (status);

Error:
	if( DAQmxFailed(error) ) {
		DAQmxGetExtendedErrorInfo(errBuff,2048);
		DAQmxClearTask(taskHandle);
		printf("DAQmx Error: %s\n",errBuff);
	}
	free(data);
	return 0;
}


void InsertSign(TaskHandle taskHandle, float sign) {
	int32       error=0;
	int32       read=0;
	char        errBuff[2048]={'\0'};

	DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,-1,10.0,DAQmx_Val_GroupByScanNumber,&data[head],BUFFER_SIZE-head,&read,NULL));
	if( read>0 ) {

		head += (read - (read%NUM_CHN));
		if (head > BUFFER_SIZE) {
			exit(5);
		}
	}
	data[head++] = 1;
	data[head++] = sign/2*senseResistorATX;
	for (int i=2; i<NUM_CHN;i++) {
		data[head++] = 0;
	}

Error:
//	printf("EveryNCallback: Inside Error\n");
	if( DAQmxFailed(error) ) {
		DAQmxGetExtendedErrorInfo(errBuff,2048);
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskHandle);
		DAQmxClearTask(taskHandle);
		printf("DAQmx Error: %s\n",errBuff);
	}
}


void TurnOffDAQ(TaskHandle taskHandle){
		int32       error=0;
		int32       read=0;
		char        errBuff[2048]={'\0'};
		int my_error = 1;

	DAQmxErrChk (DAQmxReadAnalogF64(taskHandle,-1,10.0,DAQmx_Val_GroupByScanNumber,&data[head],BUFFER_SIZE-head,&read,NULL));
	my_error = 0;
	DAQmxStopTask(taskHandle);
	DAQmxClearTask(taskHandle);
	if( read>0 ) {

//		printf("Acquired %d samples from TurnOffDAQ. Total %d\r",read,totalRead+=read);
		head += (read - (read%NUM_CHN));
		fprintf(fp, "%f\n", (float)SIGN_START);
		if (head > BUFFER_SIZE) {
			//printf ("buffer overflow!\n");
			exit(5);
		}
		fflush(stdout);

	}

	//fflush(stdout);
	unsigned long i;
	for (i = 0; i < head - NUM_CHN; i+=NUM_CHN) {
		if(fprintf(fp, "%f\n",2*data[i+1]*((data[i]+data[i+4])/senseResistorATX)+2*data[i+3]*(data[i+2]/senseResistorATX)) < 0){     
		printf ("error\n");

		}
	} 
	fprintf(fp, "%f\n",(float)SIGN_END);

Error:
	if(my_error){
		DAQmxStopTask(taskHandle);
		DAQmxClearTask(taskHandle);
     
	}
	//printf("my_error = %d\n", my_error);
	if( DAQmxFailed(error) ) {
		DAQmxGetExtendedErrorInfo(errBuff,2048);
		
		// DAQmx Stop Code
	
		printf("DAQmx Error: %s\n",errBuff);
	}
	fclose(fp);
	free(data);
	return;
}
