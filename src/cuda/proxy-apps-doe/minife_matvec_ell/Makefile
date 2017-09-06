GENCODE_SM10 := -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 := -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 := -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 := -gencode=arch=compute_62,code=\"sm_62,compute_62\"

CUOPTS =  $(GENCODE_ARCH) $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62)

EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/matvec

matvec: *.cu *.h
	nvcc main.cu -g -Xptxas -dlcm=cg  $(CUOPTS) -o $(EXECUTABLE) -Xptxas -v -lcudart

clean: 
	rm -f matvec
