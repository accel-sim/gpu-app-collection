#include "getopt.h"

char *optarg;
int optind;

extern "C"
int getopt_long (int argc, char *const *argv, const char *shortopts,
		        const struct option *longopts, int *longind)
{
	int pos = optind;
	while( pos < argc && argv[pos][0] != '-' ) pos++;
	if( pos >= argc ) return -1;
	int c = (char)argv[pos][1];
	optarg = argv[pos+1];
	optind = pos + 2;
	return c;
}