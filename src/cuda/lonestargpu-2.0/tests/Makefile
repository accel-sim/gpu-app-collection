TOPLEVEL := ../
NVCC 		:= nvcc
GCC  		:= g++

COMPUTECAPABILITY := sm_20
ifdef debug
FLAGS := -arch=$(COMPUTECAPABILITY) -g -DDEBUG=1
else
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -g -Xptxas -v
endif
INCLUDES := -I $(TOPLEVEL)/include -I ~/pd/cub-1.0.2/cub

EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)


test_sharedptr: test_sharedptr.cu  ../include/sharedptr.h
	$(NVCC) $(EXTRA) -o $@ $<