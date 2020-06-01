if [ ! -z "$1" ]
then
	ECX_MDF=$1
else
	ECX_MDF="${EINSTEIN_BUILD_DIR}/ecx/mdfs/einstein-1sm.mdf"
fi
if [ ! -z "$2" ]
then
	EMC_MM=$2
else
	EMC_MM=''
fi
#$EINSTEIN_BUILD_DIR/ecx/ecxloader --dump-knobs 1 --mdf $ECX_MDF --program rtm_0-f --stats 1

export CUDA_USE_HOST_FE=1
#setenv CUDA_USE_HOST_FE 1
export CUDA_ECX_ARGS="--dump-knobs 1 --mdf $ECX_MDF ${CUDA_ECX_ARGS} "

echo "CUDA_ECX_ARGS: ${CUDA_ECX_ARGS}"

./bin/CoMD-serial -e -x 10 -y 10 -z 10 -s 0 -m thread_atom
