GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM72 := -gencode=arch=compute_72,code=\"sm_72,compute_72\"
GENCODE_SM75 := -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
# GENCODE_SM86 := -gencode=arch=compute_86,code=\"sm_86,compute_86\"
# GENCODE_SM87 := -gencode=arch=compute_87,code=\"sm_87,compute_87\"
# GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
CUOPTS = $(GENCODE_SM70) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM80)
# CUOPTS += $(GENCODE_SM86) $(GENCODE_SM87) $(GENCODE_SM89)  


CC := nvcc

INCLUDE := 
LIB := 

SRC = sharedBankConflicts.cu

EXE = shared_bank_conflicts

NVCC_FLGAS = -Xptxas -dlcm=cv -Xptxas -dscm=wt -lineinfo

release:
	$(CC) $(NVCC_FLGAS) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

clean:
	rm -f $(EXE)

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)