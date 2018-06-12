# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# CUDA specific
LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(CUDA_PATH)/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(CUDA_LIB_PATH)

# Compiler-specific flags (by default, we always use sm_10, sm_20, and sm_30), unless we use the SMVERSION template
GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"

LANG_CUDACFLAGS=$(LANG_CFLAGS)

CFLAGS=$(APP_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS=$(APP_CXXFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS)

CUDACFLAGS=$(LANG_CUDACFLAGS) $(PLATFORM_CUDACFLAGS) $(APP_CUDACFLAGS) $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62)
CUDALDFLAGS=$(LANG_LDFLAGS) $(PLATFORM_CUDALDFLAGS) $(APP_CUDALDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: run

ifeq ($(CUDA_PATH),)
FAILSAFE=no_cuda
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(BIN) $(ARGS)

debug:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -rf $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil_cuda.o
	$(CUDALINK) $^ -o $@ $(CUDALDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil_cuda.o: $(PARBOIL_ROOT)/common/src/parboil_cuda.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(CUDACC) $< $(CUDACFLAGS) -c -o $@

no_cuda:
	@echo "CUDA_PATH is not set. Open $(CUDA_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

