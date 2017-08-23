/******************************************
 * cmdLineParser.h
 *
 * Header file for the general purpose command line parser
 * written by Sriram Swaminarayan
 *            July 24, 2007
 *****************************************/
#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_
/**
 *
 *  extern int addArg(char *longOption,char shortOption, int has_arg, char type, void *dataPtr, int dataSize, char *help);
 *
 * This call will add an argument for processing:
 *   longOption = the long name of option i.e. --optionname
 *  shortOption = the short name of option i.e. -o
 *      has_arg = whether we read an argument for this from the line i.e.
 *                is it a --longoption <value> or not
 *                if has_arg is 0, then dataPtr must be an integer pointer.
 *         type = the type of the argument. valid values are:
 *                     i   integer
 *		       f   float
 *       	       d   double
 *		       s   string
 *		       c   character
 *      dataPtr = A pointer to where the value will be stored
 *     dataSize = the length of dataPtr, only useful for character strings
 *         help = a short help string, preferably a single line or less.
 *
 *
 **/
int addArg(char *longOption,char shortOption, int has_arg, char type, void *dataPtr, int dataSize, char *help);

/**
 *   extern void processArgs(int argc, char **argv);
 *
 * Call this to process your arguments.
 * Note that this also frees any arguments added, so you can only call this once
 **/
void processArgs(int argc, char **argv);

/**
 * extern void printArgs();
 *
 * Prints the arguments to the stdout stream
 **/
extern void printArgs();

void freeArgs();

/**
 * an example

#include <stdout.h>
#include "cmdLineParser.h"

int main(int argc, char **argv) {
  int n=0;
  char infile[256]="infile value";
  char outfile[256]="outfile value";
  float fl=-1.0;
  double dbl = -1.0;
  int    flag = 0;

  //Add arguments
  addArg("infile",  'i',  1,  's',  infile,  sizeof(infile), "input file name");
  addArg("outfile", 'o',  1,  's',  outfile, sizeof(infile), "output file name");
  addArg("nSPUs",   'n',  1,  'i',  &n,      0,              "number of SPUs");
  addArg("floater",  0,   1,  'f',  &fl,     0,              "floating number");
  addArg("doubler", 'k',  1,  'd',  &dbl,    0,              "double  number");
  addArg("justFlag",'F',  0,    0,  &flag,   0,              "just a flag");

  // print the argument help
  printArgs();

  // process (and free) arguments
  processArgs(argc,argv);

  // print the variables in the code
  printf("Got n = %d\n",n);
  printf("Got fl = %f\n",fl);
  printf("Got dbl = %lf\n",dbl);
  printf("Got flag = %d\n",flag);
  printf("Got infile = %s\n",infile);
  printf("Got outfile = %s\n",outfile);

  printf("\n\n");
  
  return 0;
}

**/
#endif
