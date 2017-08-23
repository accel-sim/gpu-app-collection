ECX_MDF="${EINSTEIN_BUILD_DIR}/ecx/mdfs/einstein-1sm.mdf"
EMC_MM=''

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--dump-knobs 1 --mdf $ECX_MDF"

./CoMDCUDA -p ag -e -n 0 $@
