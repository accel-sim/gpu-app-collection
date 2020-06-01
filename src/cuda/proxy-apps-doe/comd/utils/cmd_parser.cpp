/*

Copyright (c) 2011, Los Alamos National Security, LLC All rights
reserved.  Copyright 2011. Los Alamos National Security, LLC. This
software was produced under U.S. Government contract DE-AC52-06NA25396
for Los Alamos National Laboratory (LANL), which is operated by Los
Alamos National Security, LLC for the U.S. Department of Energy. The
U.S. Government has rights to use, reproduce, and distribute this
software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF
THIS SOFTWARE.

If software is modified to produce derivative works, such modified
software should be clearly marked, so as not to confuse it with the
version available from LANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

· Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

· Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

· Neither the name of Los Alamos National Security, LLC, Los Alamos
  National Laboratory, LANL, the U.S. Government, nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS
ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/



/******************************************
 * \brief a parser for command line arguments.
 *
 * cmdLineParser.c
 *
 * A general purpose command line parser that
 * uses getopt() to parse the command line.
 *
 *
 * This file will be documented last since it is
 * not on the critical path
 *
 * written by Sriram Swaminarayan
 *            July 24, 2007
 *****************************************/
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <memory.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#define nextOption(o) ((myOption_t *) o->next)
  /**
   * \struct myOption_t the basic options struct that is used in the
   * subroutines.  This should not be visible to the programmer.
   *
   **/
typedef struct myOption_t {
  char *help;
  char *longArg;
  unsigned char  shortArg[2];
  int   argFlag;
  char  type;
  int   sz;
  void *ptr;
  void *next;
} myOption_t;

static int longest = 1;
static struct  myOption_t *myargs=NULL;

static char *dupString(char *s) {
  char *d;
  if ( ! s ) s = (char *) "";
  d = (char*)calloc(strlen(s)+1,sizeof(char));
  strcpy(d,s);
  return d;
}
  
static myOption_t *myOptionAlloc(char *longOption,char shortOption, int has_arg, char type, void *dataPtr, int dataSize, char *help) {
  myOption_t *o;
  static int iBase=129;
  o = (myOption_t*)calloc(sizeof(myOption_t),1);
  o->help = dupString(help);
  o->longArg = dupString(longOption);
  if(shortOption) o->shortArg[0] = (unsigned char)shortOption;
  else {o->shortArg[0] = iBase; iBase++;}
  o->argFlag = has_arg;
  o->type = type;
  o->ptr = dataPtr;
  o->sz = dataSize;
  if(longOption) longest = (longest>strlen(longOption)?longest:strlen(longOption));
  return o;
}

static myOption_t *myOptionFree(myOption_t *o) {
  myOption_t *r;
  if(!o) return NULL;
  r = nextOption(o);
  if(o->longArg)free(o->longArg);
  if(o->help)free(o->help);
  free(o);
  return r;
}

static myOption_t *lastOption(myOption_t *o) {
  if ( ! o) return o;
  while(nextOption(o)) o = nextOption(o);
  return o;
}

static myOption_t *findOption(myOption_t *o, unsigned char shortArg) {
  while(o) {
    if (o->shortArg[0] == shortArg) return o;
    o = nextOption(o);
  }
  return o;
}
  

int addArg(char *longOption,char shortOption, int has_arg, char type, void *dataPtr, int dataSize, char *help) {
  myOption_t *o,*p;
  o = myOptionAlloc(longOption,shortOption,has_arg,type,dataPtr,dataSize, help);
  if ( ! o ) return 1;
  if ( ! myargs) myargs = o;
  else {
    p = lastOption(myargs);
    p->next = (void *)o;
  }
  return 0;
}


void freeArgs() {
    while(myargs) {
        myargs = myOptionFree(myargs);
    } 
    return;
}

void printArgs() {
  myOption_t *o = myargs;
  char s[4096];
  unsigned char *shortArg;
  printf("\n"
	 "  Arguments are: \n");
  sprintf(s,"   --%%-%ds",longest);
  while(o) {
    if(o->shortArg[0]<0xFF) shortArg = o->shortArg;
    else shortArg = (unsigned char *) "---";
    printf(s,o->longArg);
    printf(" -%c  arg=%1d type=%c  %s\n",shortArg[0],o->argFlag,o->type,o->help);
    o = nextOption(o);
    
  }
  printf("\n\n");
  return;
}

void processArgs(int argc, char **argv) {
  myOption_t *o;
  int n=0;
  int i;
  struct option *opts;
  char *sArgs;
  int c;

  if ( ! myargs) return;
  o = myargs;
  while(o) {n++,o=nextOption(o);}

  o = myargs;
  sArgs= (char*)calloc(2*(n+2),sizeof(char));
  opts = (struct option*)calloc(n,sizeof(struct option));
  for(i=0; i<n; i++) {
    opts[i].name = o->longArg;
    opts[i].has_arg = o->argFlag;
    opts[i].flag    = 0;
    opts[i].val     = o->shortArg[0];

    strcat(sArgs,(char *) o->shortArg);
    if(o->argFlag) strcat(sArgs,":");
    o = nextOption(o);

  }

  while(1) {

    int option_index = 0;

    c = getopt_long (argc, argv, sArgs, opts, &option_index);
    if ( c == -1) break;
    o = findOption(myargs,c);
    if ( ! o ) {
      printf("\n\n"
	     "    invalid switch : -%c in getopt()\n"
	     "\n\n",
	     c);
      break;
    }      
    if(! o->argFlag) {
      int *i = (int *)o->ptr;
      *i = 1;
    }
    else {
      switch(o->type) {
      case 'i':
	sscanf(optarg,"%d",(int *)o->ptr);
	break;
      case 'f':
	sscanf(optarg,"%f",(float *)o->ptr);
	break;
      case 'd':
        sscanf(optarg,"%lf",(double *)o->ptr);
	break;
      case 's':
	strncpy((char*)o->ptr,(char*)optarg,o->sz);
	((char *)o->ptr)[o->sz-1] = '\0';
	break;
      case 'c':
	sscanf(optarg,"%c",(char *)o->ptr);
	break;
      default:
	printf("\n\n"
	       "    invalid type : %c in getopt()\n"
	       "    valid values are 'e', 'z'. 'i','d','f','s', and 'c'\n"
	       "\n\n",
	       c);      
      }
    }
  }

  free(opts);
  free(sArgs);
  //  freeArgs();
  return;
}







