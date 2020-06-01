# Compiler-specific flags (by default, we always use sm_10, sm_20, and sm_30), unless we use the SMVERSION template
GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"

all:
	nvcc -O3 ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM50} ${GENCODE_SM60} ${GENCODE_SM62} ${GENCODE_SM70} ${GENCODE_SM75} ${NVCC_ADDITIONAL_ARGS} ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe
