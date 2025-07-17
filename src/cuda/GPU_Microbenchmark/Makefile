BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin
SUB_DIRS        = $(wildcard ubench/*/*/)
SUB_DIRS_ALL    = $(SUB_DIRS:%=all-%)
SUB_DIRS_CLEAN  = $(SUB_DIRS:%=clean-%)
SUB_DIRS_TUNER = $(SUB_DIRS:%=tuner-%)


all: create_dir $(SUB_DIRS_ALL)
tuner: create_dir $(SUB_DIRS_TUNER)
clean: delete_dir $(SUB_DIRS_CLEAN)

$(SUB_DIRS_ALL):
	$(MAKE) $(MAKE_FLAGS) -C $(@:all-%=%)

$(SUB_DIRS_TUNER):
	$(MAKE) -C $(@:tuner-%=%) tuner
$(SUB_DIRS_CLEAN):
	$(MAKE) $(MAKE_FLAGS) -C $(@:clean-%=%) clean

create_dir:
	mkdir -p $(BIN_DIR)

delete_dir:
	cd $(BIN_DIR); rm -f *
