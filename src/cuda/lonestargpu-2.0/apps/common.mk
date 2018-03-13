BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++
CC := $(GCC)
CUB_DIR := $(TOPLEVEL)/cub-1.1.1

COMPUTECAPABILITY := sm_20
ifdef debug
FLAGS := -arch=$(COMPUTECAPABILITY) -g -DLSGDEBUG=1 -G
else
# including -lineinfo -G causes launches to fail because of lack of resources, pity.
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -g -Xptxas -v  #-lineinfo -G
endif
INCLUDES := -I $(TOPLEVEL)/include -I $(CUB_DIR)
LINKS := 

EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)

.PHONY: clean variants support optional-variants

ifdef APP
$(APP): $(SRC) $(INC)
	$(NVCC) $(EXTRA) -DVARIANT=0 -o $@ $<
	cp $@ $(BIN)

variants: $(VARIANTS)

optional-variants: $(OPTIONAL_VARIANTS)

support: $(SUPPORT)

clean: 
	rm -f $(APP) $(BIN)/$(APP)
ifdef VARIANTS
	rm -f $(VARIANTS)
endif
ifdef OPTIONAL_VARIANTS
	rm -f $(OPTIONAL_VARIANTS)
endif

endif
