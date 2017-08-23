if [ ! -z "$1" ]
then
	ECX_MDF=$1
else
	ECX_MDF="${EINSTEIN_BUILD_DIR}/ecx/mdfs/einstein-1tpc.mdf"
fi
if [ ! -z "$2" ]
then
	EMC_MM=$2
else
	EMC_MM=''
fi

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--dump-knobs 1 --mdf $ECX_MDF --stats 1 ${CUDA_ECX_ARGS} "

echo "Running with mdf file= $ECX_MDF"
echo "CUDA_ECX_ARGS: ${CUDA_ECX_ARGS}"

./bin/CoMD-serial -e -s 0 -x 25 -y 25 -z 25 -m thread_atom_warp_sync

