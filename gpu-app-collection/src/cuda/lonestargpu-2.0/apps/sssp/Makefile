TOPLEVEL := ../..
APP=sssp
SRC=main.cu
INC=sssp_ls.h
VARIANTS=sssp sssp-wln sssp-wlc
OTHER_VARIANTS=sssp-tpatm

include ../common.mk

sssp-wln: $(SRC) sssp_worklistn.h ../../include/worklist.h
	$(NVCC) $(EXTRA) -DVARIANT=1 -o $@ $<
	cp $@ $(BIN)

sssp-wlc: $(SRC) sssp_worklistc.h ../../include/worklistc.h
	$(NVCC) $(EXTRA) -DVARIANT=2 -o $@ $<
	cp $@ $(BIN)

