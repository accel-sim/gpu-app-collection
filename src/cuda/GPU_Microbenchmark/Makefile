BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin
ALL_SUB_DIRS    = $(wildcard ubench/*/*/)
GMMA_DIRS       = $(filter %gmma/,$(ALL_SUB_DIRS))
# ubench/wgmma/wgmma builds wgmma_verify (standalone, no CUTLASS); its CuTe
# benches are guarded inside its own Makefile, so keep it in the default build.
GMMA_DIRS      := $(filter-out ubench/wgmma/wgmma/,$(GMMA_DIRS))
SUB_DIRS        = $(filter-out $(GMMA_DIRS),$(ALL_SUB_DIRS))
SUB_DIRS_ALL    = $(SUB_DIRS:%=all-%)
SUB_DIRS_CLEAN  = $(SUB_DIRS:%=clean-%)
GMMA_DIRS_ALL   = $(GMMA_DIRS:%=all-%)
GMMA_DIRS_CLEAN = $(GMMA_DIRS:%=clean-%)


all: create_dir $(SUB_DIRS_ALL)
gmma: create_dir $(GMMA_DIRS_ALL)
clean: delete_dir $(SUB_DIRS_CLEAN)
clean_gmma: $(GMMA_DIRS_CLEAN)

$(SUB_DIRS_ALL) $(GMMA_DIRS_ALL):
	$(MAKE) $(MAKE_FLAGS) -C $(@:all-%=%)

$(SUB_DIRS_CLEAN) $(GMMA_DIRS_CLEAN):
	$(MAKE) $(MAKE_FLAGS) -C $(@:clean-%=%) clean

create_dir:
	mkdir -p $(BIN_DIR)

delete_dir:
	cd $(BIN_DIR); rm -f *
