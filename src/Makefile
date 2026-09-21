
ifneq ($(GPUAPPS_SETUP_ENVIRONMENT_WAS_RUN),1)
$(error You must run "source setup_environment" before calling make)
endif

ifeq ($(CUDA_GT_7), 1)
all:   pannotia rodinia_2.0-ft proxy-apps microbench rodinia-3.1 ispass-2009 lonestargpu-2.0 polybench parboil shoc custom_apps deeplearning cutlass GPU_Microbenchmark heterosync Deepbench_nvidia
else
ifeq ($(CUDA_GT_4), 1)
all:   pannotia rodinia_2.0-ft proxy-apps microbench rodinia-3.1 ispass-2009 lonestargpu-2.0 polybench parboil shoc custom_apps
else
all:   pannotia rodinia_2.0-ft proxy-apps microbench rodinia-3.1 ispass-2009 polybench parboil shoc custom_apps
endif
endif

#Disable clean for now, It has a bug!
# clean_dragon-naive clean_pannotia clean_proxy-apps
clean: clean_rodinia_2.0-ft clean_dragon-cdp  clean_ispass-2009 clean_lonestargpu-2.0 clean_custom_apps clean_parboil clean_cutlass clean_rodinia-3.1 clean_heterosync

clean_data:
	./clean_data.sh

data:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	cd ../ && bash ./get_data.sh

###################################################################################################3
# Rodinia 2.0 Functional Test Stuff
###################################################################################################3
rodinia_2.0-ft:
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/backprop
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/bfs
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/heartwall
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/hotspot
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/kmeans
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/lud
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/nn
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/nw
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/pathfinder
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/srad
	$(SETENV) make $(MAKE_ARGS) -C cuda/rodinia/2.0-ft/streamcluster

###################################################################################################3
# Purdue microbenchmarks for added functionality
###################################################################################################3
microbench:
	$(SETENV) make $(MAKE_ARGS) -C cuda/microbench cuda-$(CUDA_VERSION_MAJOR)

###################################################################################################3
# For Dragon, we need to change the archs manually!  (TO DO)
# Naive works wwith only SM_20 and above
# Fro Dragon-cdp comilation, you need to ensure that you are using at least CUDA 5.5
###################################################################################################3
# NOTE: dragon-naive/dragon-cdp are omitted from "all" - dragon_li needs boost
# dev headers and only knows sm_20/35/50/61, all removed in CUDA 12/13.
# Still buildable explicitly via "make dragon-naive" once those are addressed.
dragon-naive: 
	chmod +x cuda/dragon_li/sconstruct
	if [ ${CUDA_VERSION_MAJOR} -lt 8 ]; then \
		scons sm35=1 no_debug=1 -C cuda/dragon_li; \
	else \
		scons sm35=1 sm61=1 no_debug=1 -C cuda/dragon_li; \
	fi
	cp ./cuda/dragon_li/bin/$(CUDA_VERSION)/testAmr $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/bin/$(CUDA_VERSION)/testBfs $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/bin/$(CUDA_VERSION)/testJoin $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/bin/$(CUDA_VERSION)/testSssp $(BINDIR)/$(BINSUBDIR)/

dragon-cdp: dragon-naive
	chmod +x cuda/dragon_li/sconstruct
	if [ ${CUDA_VERSION_MAJOR} -lt 8 ]; then \
		scons cdp=1 no_debug=1 sm35=1 -C cuda/dragon_li; \
	else \
		scons cdp=1 no_debug=1 sm61=1 sm35=1 -C cuda/dragon_li; \
	fi
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testAmr-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testBfs-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testJoin-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testSssp-cdp $(BINDIR)/$(BINSUBDIR)/

###################################################################################################3
#Microbenchmarks for cache
###################################################################################################3

GPU_Microbenchmark:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	$(SETENV) make $(MAKE_ARGS) -C cuda/GPU_Microbenchmark
	cp -r cuda/GPU_Microbenchmark/bin/* $(BINDIR)/$(BINSUBDIR)/


# conv_bench and rnn_bench are disabled: they use 13 cuDNN APIs that cuDNN 9
# removed (the whole v7 RNN path, plus the heuristic
# cudnnGetConvolution*Algorithm calls). Porting them changes which kernels
# the benchmarks actually run, so it needs a deliberate migration rather
# than a build fix. gemm_bench is unaffected.
Deepbench_nvidia:
	$(SETENV) make $(MAKE_ARGS) -C cuda/DeepBench/code/nvidia gemm
	cp -r cuda/DeepBench/code/nvidia/bin/gemm_bench* $(BINDIR)/$(BINSUBDIR)/

###################################################################################################3
#pagerank and bc does not work with SM_10 because they need atomic_add
###################################################################################################3

pannotia:
	$(SETENV) make $(MAKE_ARGS) -C cuda/pannotia/bc
	$(SETENV) export VARIANT="MAX"; make $(MAKE_ARGS) -C cuda/pannotia/color
	$(SETENV) export VARIANT="MAXMIN"; make $(MAKE_ARGS) -C cuda/pannotia/color
	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) -C cuda/pannotia/fw
	$(SETENV) export VARIANT="BLOCK"; make $(MAKE_ARGS) -C cuda/pannotia/fw
	$(SETENV) make $(MAKE_ARGS) -C cuda/pannotia/mis
	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="SPMV"; make $(MAKE_ARGS) -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="CSR"; make $(MAKE_ARGS) -C cuda/pannotia/sssp
	$(SETENV) export VARIANT="ELL"; make $(MAKE_ARGS) -C cuda/pannotia/sssp

###################################################################################################3
#TO DO
#note: matvec does not work with cuda 8.0
#comd does not wark with cuda 4.2
#xsbench does not work with SM_10
###################################################################################################3


proxy-apps:
	chmod +x cuda/proxy-apps-doe/cns/compile.bash
	($(SETENV) cd cuda/proxy-apps-doe/cns/ ; ./compile.bash)
	cp cuda/proxy-apps-doe/cns/cns_all $(BINDIR)/$(BINSUBDIR)/
	#chmod +x cuda/proxy-apps-doe/comd/cmd_compile.sh
	#( cd cuda/proxy-apps-doe/comd ; ./cmd_compile.sh)
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/lulesh
	if [ ${CUDA_VERSION_MAJOR} -lt 7 ] ; then  \
		$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/minife_matvec_ell;\
	fi
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/xsbench

rodinia-3.1:
	if [ ${CUDA_VERSION_MAJOR} -gt 5 ]; then \
		$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/huffman/; \
	fi
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/backprop -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/bfs -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/cfd -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hotspot -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/kmeans -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/nw -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/streamcluster -f Makefile_nvidia
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/mummergpu
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/b+tree/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/dwt2d/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/heartwall/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hybridsort/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/myocyte/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/nn/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/particlefilter/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/particlefilter/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/pathfinder/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/lavaMD/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/lud/cuda/
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/leukocyte/CUDA/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hotspot3D/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/gaussian
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/srad/srad_v1
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/srad/srad_v2 -f Makefile_nvidia
#	mv cuda/rodinia/3.1/cuda/leukocyte/CUDA/leukocyte $(BINDIR)/$(BINSUBDIR)/leukocyte-rodinia-3.1

ispass-2009:
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/AES
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/BFS
#	cd CP; export PARBOIL_ROOT=`pwd`; cd common/src; make $(MAKE_ARGS); cd -; ./parboil compile cp cuda_short; cp benchmarks/cp/build/cuda_short/cp $(BINDIR)/$(BINSUBDIR)/CP 
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG/3rdParty/ParMetis-3.1
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/LIB
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/LPS
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/MUM
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/NN
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/NQU
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/RAY
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/STO
# WP is Fortran and needs gfortran, which is not installed. Re-enable this
# line after "apt install gfortran". Note the recipe leaves a WP-$PID copy
# behind whenever the inner make fails.
#	PID=$$$$ && cp -r cuda/ispass-2009/WP cuda/ispass-2009/WP-$$PID && $(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/ispass-2009/WP-$$PID && rm -rf cuda/ispass-2009/WP-$$PID

lonestargpu-2.0:
	$(setenv) make $(make_args) noinline=$(noinline) -C cuda/lonestargpu-2.0 all
	mv cuda/lonestargpu-2.0/apps/bfs/bfs $(BINDIR)/$(BINSUBDIR)/lonestar-bfs
	mv cuda/lonestargpu-2.0/apps/bfs/bfs-atomic $(BINDIR)/$(BINSUBDIR)/lonestar-bfs-atomic
	mv cuda/lonestargpu-2.0/apps/bfs/bfs-wlc $(BINDIR)/$(BINSUBDIR)/lonestar-bfs-wlc
	mv cuda/lonestargpu-2.0/apps/bfs/bfs-wla $(BINDIR)/$(BINSUBDIR)/lonestar-bfs-wla
	mv cuda/lonestargpu-2.0/apps/bfs/bfs-wlw $(BINDIR)/$(BINSUBDIR)/lonestar-bfs-wlw
	mv cuda/lonestargpu-2.0/apps/bh/bh $(BINDIR)/$(BINSUBDIR)/lonestar-bh
	mv cuda/lonestargpu-2.0/apps/dmr/dmr $(BINDIR)/$(BINSUBDIR)/lonestar-dmr
	mv cuda/lonestargpu-2.0/apps/mst/mst $(BINDIR)/$(BINSUBDIR)/lonestar-mst
	mv cuda/lonestargpu-2.0/apps/pta/pta $(BINDIR)/$(BINSUBDIR)/lonestar-pta
	mv cuda/lonestargpu-2.0/apps/sp/nsp $(BINDIR)/$(BINSUBDIR)/lonestar-nsp
	mv cuda/lonestargpu-2.0/apps/sssp/sssp $(BINDIR)/$(BINSUBDIR)/lonestar-sssp
	mv cuda/lonestargpu-2.0/apps/sssp/sssp-wlc $(BINDIR)/$(BINSUBDIR)/lonestar-sssp-wlc
	mv cuda/lonestargpu-2.0/apps/sssp/sssp-wln $(BINDIR)/$(BINSUBDIR)/lonestar-sssp-wln

parboil:
	make data
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	$(SETENV) cd cuda/parboil; ./parboil compile cutcp cuda
	$(SETENV) cd cuda/parboil; ./parboil compile bfs cuda
	$(SETENV) cd cuda/parboil; ./parboil compile histo cuda
	$(SETENV) cd cuda/parboil; ./parboil compile lbm cuda
	$(SETENV) cd cuda/parboil; ./parboil compile mri-gridding cuda
	$(SETENV) cd cuda/parboil; ./parboil compile mri-q cuda
	$(SETENV) cd cuda/parboil; ./parboil compile sad cuda
	$(SETENV) cd cuda/parboil; ./parboil compile sgemm cuda
	$(SETENV) cd cuda/parboil; ./parboil compile spmv cuda
	$(SETENV) cd cuda/parboil; ./parboil compile stencil cuda
	$(SETENV) cd cuda/parboil; ./parboil compile tpacf cuda
	mv ./cuda/parboil/benchmarks/lbm/build/cuda_default/lbm $(BINDIR)/$(BINSUBDIR)/parboil-lbm
	mv ./cuda/parboil/benchmarks/cutcp/build/cuda_default/cutcp $(BINDIR)/$(BINSUBDIR)/parboil-cutcp
	mv ./cuda/parboil/benchmarks/bfs/build/cuda_default/bfs $(BINDIR)/$(BINSUBDIR)/parboil-bfs
	mv ./cuda/parboil/benchmarks/histo/build/cuda_default/histo $(BINDIR)/$(BINSUBDIR)/parboil-histo
	mv ./cuda/parboil/benchmarks/mri-gridding/build/cuda_default/mri-gridding $(BINDIR)/$(BINSUBDIR)/parboil-mri-gridding
	mv ./cuda/parboil/benchmarks/mri-q/build/cuda_default/mri-q $(BINDIR)/$(BINSUBDIR)/parboil-mri-q
	mv ./cuda/parboil/benchmarks/sad/build/cuda_default/sad $(BINDIR)/$(BINSUBDIR)/parboil-sad
	mv ./cuda/parboil/benchmarks/sgemm/build/cuda_default/sgemm $(BINDIR)/$(BINSUBDIR)/parboil-sgemm
	mv ./cuda/parboil/benchmarks/spmv/build/cuda_default/spmv $(BINDIR)/$(BINSUBDIR)/parboil-spmv
	mv ./cuda/parboil/benchmarks/stencil/build/cuda_default/stencil $(BINDIR)/$(BINSUBDIR)/parboil-stencil
	mv ./cuda/parboil/benchmarks/tpacf/build/cuda_default/tpacf $(BINDIR)/$(BINSUBDIR)/parboil-tpacf

polybench:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	$(SETENV) cd cuda/polybench-gpu-1.0/CUDA/; sh ./compileCodes.sh "$(MAKE_ARGS)"
	mv cuda/polybench-gpu-1.0/CUDA/2DCONV/2DConvolution.exe $(BINDIR)/$(BINSUBDIR)/polybench-2DConvolution
	mv cuda/polybench-gpu-1.0/CUDA/2MM/2mm.exe $(BINDIR)/$(BINSUBDIR)/polybench-2mm
	mv cuda/polybench-gpu-1.0/CUDA/3DCONV/3DConvolution.exe $(BINDIR)/$(BINSUBDIR)/polybench-3DConvolution
	mv cuda/polybench-gpu-1.0/CUDA/3MM/3mm.exe $(BINDIR)/$(BINSUBDIR)/polybench-3mm
	mv cuda/polybench-gpu-1.0/CUDA/ATAX/atax.exe $(BINDIR)/$(BINSUBDIR)/polybench-atax
	mv cuda/polybench-gpu-1.0/CUDA/BICG/bicg.exe $(BINDIR)/$(BINSUBDIR)/polybench-bicg
	mv cuda/polybench-gpu-1.0/CUDA/CORR/correlation.exe $(BINDIR)/$(BINSUBDIR)/polybench-correlation
	mv cuda/polybench-gpu-1.0/CUDA/COVAR/covariance.exe $(BINDIR)/$(BINSUBDIR)/polybench-covariance
	mv cuda/polybench-gpu-1.0/CUDA/FDTD-2D/fdtd2d.exe $(BINDIR)/$(BINSUBDIR)/polybench-fdtd2d
	mv cuda/polybench-gpu-1.0/CUDA/GEMM/gemm.exe $(BINDIR)/$(BINSUBDIR)/polybench-gemm
	mv cuda/polybench-gpu-1.0/CUDA/GESUMMV/gesummv.exe $(BINDIR)/$(BINSUBDIR)/polybench-gesummv
	mv cuda/polybench-gpu-1.0/CUDA/GRAMSCHM/gramschmidt.exe $(BINDIR)/$(BINSUBDIR)/polybench-gramschmidt
	mv cuda/polybench-gpu-1.0/CUDA/MVT/mvt.exe $(BINDIR)/$(BINSUBDIR)/polybench-mvt
	mv cuda/polybench-gpu-1.0/CUDA/SYR2K/syr2k.exe $(BINDIR)/$(BINSUBDIR)/polybench-syr2k
	mv cuda/polybench-gpu-1.0/CUDA/SYRK/syrk.exe $(BINDIR)/$(BINSUBDIR)/polybench-syrk

shoc:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	cd cuda/shoc-master/; ./configure; $(SETENV) make $(MAKE_ARGS); $(SETENV) make $(MAKE_ARGS) -C src/cuda
	mv cuda/shoc-master/src/cuda/level0/BusSpeedDownload $(BINDIR)/$(BINSUBDIR)/shoc-BusSpeedDownload
	mv cuda/shoc-master/src/cuda/level0/BusSpeedReadback $(BINDIR)/$(BINSUBDIR)/shoc-BusSpeedReadback
	mv cuda/shoc-master/src/cuda/level0/DeviceMemory $(BINDIR)/$(BINSUBDIR)/shoc-DeviceMemory
	mv cuda/shoc-master/src/cuda/level0/MaxFlops $(BINDIR)/$(BINSUBDIR)/shoc-MaxFlops
	mv cuda/shoc-master/src/cuda/level1/bfs/BFS $(BINDIR)/$(BINSUBDIR)/shoc-BFS
	mv cuda/shoc-master/src/cuda/level1/fft/FFT $(BINDIR)/$(BINSUBDIR)/shoc-FFT
	mv cuda/shoc-master/src/cuda/level1/gemm/GEMM $(BINDIR)/$(BINSUBDIR)/shoc-GEMM
	mv cuda/shoc-master/src/cuda/level1/md/MD $(BINDIR)/$(BINSUBDIR)/shoc-MD
	mv cuda/shoc-master/src/cuda/level1/md5hash/MD5Hash $(BINDIR)/$(BINSUBDIR)/shoc-MD5Hash
	mv cuda/shoc-master/src/cuda/level1/neuralnet/NeuralNet $(BINDIR)/$(BINSUBDIR)/shoc-NeuralNet
	mv cuda/shoc-master/src/cuda/level1/reduction/Reduction $(BINDIR)/$(BINSUBDIR)/shoc-Reduction
	mv cuda/shoc-master/src/cuda/level1/scan/Scan $(BINDIR)/$(BINSUBDIR)/shoc-Scan
	mv cuda/shoc-master/src/cuda/level1/sort/Sort $(BINDIR)/$(BINSUBDIR)/shoc-Sort
	mv cuda/shoc-master/src/cuda/level1/spmv/Spmv $(BINDIR)/$(BINSUBDIR)/shoc-Spmv
	mv cuda/shoc-master/src/cuda/level1/stencil2d/Stencil2D $(BINDIR)/$(BINSUBDIR)/shoc-Stencil2D
	mv cuda/shoc-master/src/cuda/level1/triad/Triad $(BINDIR)/$(BINSUBDIR)/shoc-Triad
	mv cuda/shoc-master/src/cuda/level2/qtclustering/QTC $(BINDIR)/$(BINSUBDIR)/shoc-QTC
	mv cuda/shoc-master/src/cuda/level2/s3d/S3D $(BINDIR)/$(BINSUBDIR)/shoc-S3D
	mv cuda/shoc-master/src/stability/Stability $(BINDIR)/$(BINSUBDIR)/shoc-Stability

custom_apps:
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/custom-apps/shoc-modified-spmv/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/custom-apps/rodinia-kmn-no-tex/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/custom-apps/sdk-matrixMul-modified/

power:
	$(SETENV) make $(MAKE_ARGS) PWRTYPE=SM noinline=$(noinline) -C cuda/gpuwattch-ubench/ power
	$(SETENV) make $(MAKE_ARGS) PWRTYPE=HW noinline=$(noinline) -C cuda/gpuwattch-ubench/ power

deeplearning:
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/cudnn/mnist
	cp cuda/cudnn/mnist/mnistCUDNN $(BINDIR)/$(BINSUBDIR)/

cutlass:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	git submodule init && git submodule update
	$(SETENV) mkdir -p cuda/cutlass-bench/build && cd cuda/cutlass-bench/build && cmake .. -DUSE_GPGPUSIM=1 -DCUTLASS_NVCC_ARCHS=75 && make cutlass_perf_test
	cd cuda/cutlass-bench/build/tools/test/perf && ln -s -f ../../../../binary.sh . && ./binary.sh
	cp cuda/cutlass-bench/build/tools/test/perf/cutlass_perf_test $(BINDIR)/$(BINSUBDIR)/

# Maybe we should use submodules for this - but I have heard a lot of horor stories about these..
# For now - lets just clone if we don't have it and set the SHA we want.
heterosync:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	cd cuda && \
	if [ ! -d "heterosync" ]; then \
		git clone git@github.com:mattsinc/heterosync.git; \
	fi && \
	cd heterosync && git checkout 22bc0eb
# heterosync is an external clone (gitignored), so its Makefile is overridden
# from here rather than edited: it hardcodes sm_60 (removed in CUDA 13) and
# only looks for lib64 when uname -m is x86_64, and calls cudaThreadSynchronize
# which CUDA 13 removed.
	$(SETENV) make $(MAKE_ARGS) CUDA_DIR=$(CUDA_INSTALL_PATH) \
		CUDA_LIB_DIR=$(CUDA_LIB_DIR) \
		OPTS="-O3 $(CUDA_CPPFLAGS) -DcudaThreadSynchronize=cudaDeviceSynchronize" \
		-C cuda/heterosync/cuda/syncPrims/uvm/
	mv cuda/heterosync/cuda/syncPrims/uvm/allSyncPrims-1kernel $(BINDIR)/$(BINSUBDIR)/

clean_heterosync:
	rm -rf cuda/heterosync

clean_cutlass:
	rm -fr cuda/cutlass-bench/build

clean_deeplearning:
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/cudnn/mnist clean

clean_custom_apps:
	make clean -C cuda/custom-apps/shoc-modified-spmv/
	make clean -C cuda/custom-apps/rodinia-kmn-no-tex/
	make clean -C cuda/custom-apps/sdk-matrixMul-modified/

clean_shoc:
	cd cuda/shoc-master/; make clean; make distclean

clean_parboil:
	$(SETENV) cd cuda/parboil; ./parboil clean cutcp cuda
	$(SETENV) cd cuda/parboil; ./parboil clean bfs cuda
	$(SETENV) cd cuda/parboil; ./parboil clean histo cuda
	$(SETENV) cd cuda/parboil; ./parboil clean lbm cuda
	$(SETENV) cd cuda/parboil; ./parboil clean mri-gridding cuda
	$(SETENV) cd cuda/parboil; ./parboil clean mri-q cuda
	$(SETENV) cd cuda/parboil; ./parboil clean sad cuda
	$(SETENV) cd cuda/parboil; ./parboil clean sgemm cuda
	$(SETENV) cd cuda/parboil; ./parboil clean spmv cuda
	$(SETENV) cd cuda/parboil; ./parboil clean stencil cuda
	$(SETENV) cd cuda/parboil; ./parboil clean tpacf cuda

clean_lonestargpu-2.0:
	$(setenv) make $(make_args) noinline=$(noinline) -C cuda/lonestargpu-2.0 clean

clean_ispass-2009:
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/AES
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/BFS
#	cd CP; export PARBOIL_ROOT=`pwd`; cd common/src; make $(MAKE_ARGS); cd -; ./parboil compile cp cuda_short; cp benchmarks/cp/build/cuda_short/cp $(BINDIR)/$(BINSUBDIR)/CP 
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG/3rdParty/ParMetis-3.1
#	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C DG
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/LIB
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/LPS
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/MUM
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/NN
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/NQU
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/RAY
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/STO
	$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/ispass-2009/WP

clean_rodinia-3.1:
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/backprop -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/bfs -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/cfd -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hotspot -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/kmeans -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/nw -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/streamcluster -f Makefile_nvidia
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/mummergpu
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/b+tree/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/dwt2d/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/heartwall/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/huffman/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hybridsort/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/myocyte/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/nn/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/particlefilter/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/particlefilter/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/pathfinder/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/lavaMD/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/lud/cuda/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/leukocyte/CUDA/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hotspot3D/
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/gaussian
	$(SETENV) make clean $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/srad/

clean_rodinia_2.0-ft:
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/backprop
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/bfs
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/heartwall
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/hotspot
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/kmeans
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/lud
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/nn
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/nw
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/pathfinder
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/srad
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/rodinia/2.0-ft/streamcluster

clean_dragon-naive: 
	$(SETENV) rm -f /cuda/dragon_li/bin

clean_dragon-cdp: 
	$(SETENV) rm -f /cuda/dragon_li/cdp_bin

clean_pannotia: 
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/pannotia/bc
	$(SETENV) export VARIANT="MAX"; make $(MAKE_ARGS) clean -C cuda/pannotia/color
	$(SETENV) export VARIANT="MAXMIN"; make $(MAKE_ARGS) clean -C cuda/pannotia/color
	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C cuda/pannotia/fw
	$(SETENV) export VARIANT="BLOCK"; make $(MAKE_ARGS) clean -C cuda/pannotia/fw
	$(SETENV) make $(MAKE_ARGS)  -C cuda/pannotia/mis
	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="SPMV"; make $(MAKE_ARGS) clean -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="DEFAULT"; make $(MAKE_ARGS) clean -C cuda/pannotia/sssp
	$(SETENV) export VARIANT="ELL"; make $(MAKE_ARGS) clean -C cuda/pannotia/sssp

clean_proxy-apps:
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/proxy-apps-doe/lulesh
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/proxy-apps-doe/minife_matvec_ell
	$(SETENV) make $(MAKE_ARGS) clean -C cuda/proxy-apps-doe/xsbench
	chmod +x cuda/proxy-apps-doe/cns/compile.bash
	(cd cuda/proxy-apps-doe/cns/ ; ./compile.bash -c)
	chmod +x cuda/proxy-apps-doe/comd/clean.sh
	( cd cuda/proxy-apps-doe/comd ; ./clean.sh ) 
