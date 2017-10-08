ifeq ($(GPGPUSIM_BENCHMARKS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

all:
	nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) --cudart shared oo-vfunc-test.cu -o $(BINDIR)/$(BINSUBDIR)/microbench-oo-vfunc-test
