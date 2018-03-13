#include <stdio.h>
#include <stdlib.h>

unsigned NCLAUSES, NLITERALS, LITPERCLA;

static unsigned *lperc;

bool inclause(unsigned newlit, unsigned iilperc) {
  for (unsigned ii = 0; ii < iilperc; ++ii) {
    if (lperc[ii] == newlit) {
      return true;
    }
  }
  return false;
}

unsigned getnextliteral() {
  static unsigned iilperc = 0;

  if (iilperc == LITPERCLA) {
    iilperc = 0;
  }
  unsigned newlit;
  do {
    newlit = rand() % NLITERALS;
  } while (inclause(newlit, iilperc));
  lperc[iilperc++] = newlit;
  return newlit;
}

void dump_formula(FILE *of, unsigned *c2l, bool *eisneg)
{
  fprintf(of, "p cnf %d %d\n", NLITERALS, NCLAUSES);

  for(int mm = 0, row = 0; mm < NCLAUSES; mm++, row+=LITPERCLA) {

    for(int kk = 0; kk < LITPERCLA; kk++) {
      fprintf(of, "%d ", eisneg[row+kk] ? -(c2l[row+kk]+1) : (c2l[row+kk]+1));
    }

    fprintf(of, "0\n");
  }
}

void init(unsigned *c2l, bool *eisneg) {
  unsigned mm, nn, kk;

  // init randomly
  lperc = (unsigned *)malloc(LITPERCLA * sizeof(unsigned));
  for (mm = 0; mm < NCLAUSES; ++mm) {
    unsigned row = mm * LITPERCLA;
    for (kk = 0; kk < LITPERCLA; ++kk) {
      unsigned newlit = getnextliteral();
      c2l[row + kk] = newlit;
      eisneg[row + kk] = (bool)(rand() % 2);		// init.
    }
  }

  free(lperc);
}

int main(int argc, char *argv[])
{
  unsigned int hnedges;
  unsigned *hc2l;
  bool *heisneg;

  if (argc != 5 && argc != 6) {
    printf("Usage: %s seed M N K [file]\n", argv[0]);
    exit(1);
  }

  unsigned argno = 0;

  srand(atoi(argv[++argno]));		// seed.

  NCLAUSES = atoi(argv[++argno]);		// M.
  NLITERALS = atoi(argv[++argno]);	// N.
  LITPERCLA = atoi(argv[++argno]);	// K.

  const char *OUTFILE = NULL;
  if(argc == 6)
    OUTFILE = argv[++argno];

  FILE *output;

  if(OUTFILE)
    output = fopen(OUTFILE, "w");
  else
    output = stdout;

  hnedges = NCLAUSES * LITPERCLA;

  hc2l = (unsigned *)malloc(hnedges * sizeof(unsigned));
  heisneg = (bool *)malloc(hnedges * sizeof(bool));

  init(hc2l, heisneg);

  dump_formula(output, hc2l, heisneg);

  return 0;
}
