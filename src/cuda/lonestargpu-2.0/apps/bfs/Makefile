TOPLEVEL := ../..
APP=bfs
SRC=main.cu 
INC=bfs_ls.h
B40_DIR=./back40computing-read-only/
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test/ 
VARIANTS=bfs bfs-atomic bfs-wlc
OPTIONAL_VARIANTS=bfs-merrill bfs-wlw bfs-wla

include ../common.mk

bfs-atomic: $(SRC) bfs_topo_atomic.h
	$(NVCC) $(EXTRA) -DVARIANT=1 -o $@ $<
	cp $@ $(BIN)

bfs-merrill: $(SRC) bfs_merrill.h $(B40_DIR)
	$(NVCC) $(EXTRA) $(B40C_INC) -DVARIANT=2 -o $@ $<
	cp $@ $(BIN)

bfs-wlw: $(SRC) bfs_worklistw.h ../../include/worklist.h
	$(NVCC) $(EXTRA) $(B40C_INC) -DVARIANT=3 -o $@ $<
	cp $@ $(BIN)

bfs-wla: $(SRC) bfs_worklista.h
	$(NVCC) $(EXTRA) -DVARIANT=5 -o $@ $<
	cp $@ $(BIN)

bfs-wlc: $(SRC) bfs_worklistc.h ../../include/worklistc.h
	$(NVCC) $(EXTRA) -DVARIANT=6 -o $@ $<
	cp $@ $(BIN)
