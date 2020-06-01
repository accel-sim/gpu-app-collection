#include "XSbench_header.h"

// Initializes papi counters. I'm still working on getting PAPI counters
// fully integrated. As is now, I've been commenting in/out the particular
// counters I want at any given time.
void counter_init( int * eventset, int * num_papi_events )
{
	printf("Initializing PAPI counters...\n");
	int names[] = 
	{
		/*
		PAPI_TOT_INS,
		PAPI_INT_INS,
		PAPI_FP_INS,
		PAPI_LD_INS,
		PAPI_SR_INS,
		PAPI_BR_INS,
		PAPI_VEC_INS,
		PAPI_RES_STL, // All Resource Stalls
		PAPI_FP_STAL, // Floating Point stalls (bad)
		PAPI_TOT_CYC, // Total cycles
		PAPI_LST_INS,
		PAPI_SYC_INS,
		PAPI_L2_DCH, // Level 2 Data cache Hits
		PAPI_L2_DCA // L2 Data cache Accesses
		PAPI_L3_DCA // L3 Data cache Accesses
		PAPI_L2_DCR,
		PAPI_L3_DCR,
		PAPI_L2_DCW,
		PAPI_L3_DCW,
		PAPI_L2_ICH,
		PAPI_L3_ICH,
		PAPI_L2_ICA,
		PAPI_L3_ICA,
		PAPI_L2_ICR,
		PAPI_L3_ICR,
		PAPI_L2_ICW,
		PAPI_L3_ICW,
		*/
		//PAPI_L2_TCH,
		//PAPI_L2_TCA,
		//PAPI_L3_TCA
		//PAPI_L2_TCR,
		//PAPI_L3_TCR,
		//PAPI_L2_TCW
		/*
		PAPI_L3_TCW,
		PAPI_FML_INS,
		PAPI_FAD_INS,
		PAPI_FDV_INS,
		PAPI_FSQ_INS,
		PAPI_FNV_INS,
		PAPI_FP_OPS,
		PAPI_SP_OPS,
		PAPI_DP_OPS,
		PAPI_VEC_SP,
		PAPI_VEC_DP,
		PAPI_REF_CYC,
		*/
		//PAPI_L2_TCM,
		//PAPI_L2_TCA,
		//PAPI_L3_TCM,
		//PAPI_L3_TCA
		PAPI_TLB_TL,
		PAPI_TOT_CYC
	};

	*num_papi_events = sizeof(names) / sizeof(int);
	
	PAPI_library_init(83886080);

	PAPI_create_eventset(eventset);

	for( int i = 0; i < *num_papi_events; i++ )
		PAPI_add_event( *eventset, names[i] );

	//PAPI_add_events( *eventset, names, NUM_PAPI_EVENTS );

	PAPI_start(*eventset);
}

// Stops the papi counters and prints results
void counter_stop( int * eventset, int num_papi_events )
{
	int * events = malloc(num_papi_events * sizeof(int));
	int n = num_papi_events;
	PAPI_list_events( *eventset, events, &n );
	PAPI_event_info_t info;
	
	long_long * values = malloc( num_papi_events * sizeof(long_long));

	PAPI_stop(*eventset, values);

	FILE * out = fopen("counters.txt", "a");
	if( PRINT_PAPI_INFO )
	{	
		center_print("PAPI INFORMATION", 79);
		border_print();
		printf("Count          \tSmybol      \tDescription\n");
	}
	for( int i = 0; i < num_papi_events; i++ )
	{
		PAPI_get_event_info(events[i], &info);
		fprintf(out,"%lld\t",values[i]);
		if( PRINT_PAPI_INFO )
			printf("%-15lld\t%s\t%s\n", values[i],info.symbol,info.long_descr);
	}
	fprintf(out,"\n");
	if( PRINT_PAPI_INFO )
		border_print();
	free(events);
	free(values);	
	fclose(out);
}
