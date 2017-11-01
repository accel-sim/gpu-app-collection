
ifeq ($(GPGPUSIM_BENCHMARKS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive microbench rodinia-3.1 sdk-4.2
#  dragon-cdp 

#Disable clean for now, It has a bug!
#clean: clean_rodinia_2.0-ft 
#	../data_dirs/clean_data.sh

#clean_dragon-naive clean_dragon-cdp clean_pannotia clean_proxy-apps

data:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	bash ../data_dirs/get_data.sh

###################################################################################################3
# Rodinia 2.0 Functional Test Stuff
###################################################################################################3
rodinia_2.0-ft: data
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
dragon-naive: data
	chmod +x cuda/dragon_li/sconstruct
	if [ ${CUDA_VERSION_MAJOR} -lt 8 ]; then \
		scons sm20=1 -C cuda/dragon_li; \
	else \
		scons sm62=1 -C cuda/dragon_li; \
	fi

dragon-cdp: data
	chmod +x cuda/dragon_li/sconstruct
	scons cdp=1 -C cuda/dragon_li
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testAmr-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testBfs-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testJoin-cdp $(BINDIR)/$(BINSUBDIR)/
	cp ./cuda/dragon_li/cdp_bin/$(CUDA_VERSION)/testSssp-cdp $(BINDIR)/$(BINSUBDIR)/

###################################################################################################3
#pagerank and bc does not work with SM_10 because they need atomic_add
###################################################################################################3

pannotia: data
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


proxy-apps: data
	chmod +x cuda/proxy-apps-doe/cns/compile.bash
	($(SETENV) cd cuda/proxy-apps-doe/cns/ ; ./compile.bash)
	#chmod +x cuda/proxy-apps-doe/comd/cmd_compile.sh
	#( cd cuda/proxy-apps-doe/comd ; ./cmd_compile.sh) 
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/lulesh
	if [ ${CUDA_VERSION_MAJOR} -lt 7 ] ; then  \
		$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/minife_matvec_ell;\
	fi
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/xsbench

SDK_42_APPS=$(shell ls -p cuda/sdk/4.2/ | grep "/")
$(SDK_42_APPS):
	$(SETENV) export LIBDIRSDK=$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/lib; export COMMONDIR=$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/common; export SHAREDDIR=$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/shared; export OMIT_CUTIL_LIB=0; export OMIT_SHRUTIL_LIB=0; make $(MAKE_ARGS) noinline=$(noinline) -C cuda/sdk/4.2/$@

sdk-4.2: $(SDK_42_APPS)

RODINIA_31_APPS=$(shell ls -p cuda/rodinia/3.1/cuda | grep "/")
$(RODINIA_31_APPS):
	if [ -f cuda/rodinia/3.1/cuda/$@/Makefile_nvidia ]; then \
		$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@ -f Makefile_nvidia; \
	elif [ -f cuda/rodinia/3.1/cuda/$@/CUDA/Makefile ]; then \
		$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@/CUDA; \
	else \
		$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@; \
	fi

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
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/leukocyte/CUDA/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/hotspot3D/
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/gaussian
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/rodinia/3.1/cuda/srad/
	if [ ${CUDA_VERSION_MAJOR} -gt 5 ]; then \
		mv cuda/rodinia/3.1/cuda/huffman/pavle $(BINDIR)/$(BINSUBDIR)/huffman-rodinia-3.1; \
	fi
	mv cuda/rodinia/3.1/cuda/b+tree/b+tree.out $(BINDIR)/$(BINSUBDIR)/b+tree.out-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/dwt2d/dwt2d $(BINDIR)/$(BINSUBDIR)/dwt2d-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/heartwall/heartwall $(BINDIR)/$(BINSUBDIR)/heartwall-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/hybridsort/hybridsort $(BINDIR)/$(BINSUBDIR)/hybridsort-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/myocyte/myocyte.out $(BINDIR)/$(BINSUBDIR)/myocyte.out-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/nn/nn $(BINDIR)/$(BINSUBDIR)/nn-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/particlefilter/particlefilter_float $(BINDIR)/$(BINSUBDIR)/particlefilter_float-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/particlefilter/particlefilter_naive $(BINDIR)/$(BINSUBDIR)/particlefilter_naive-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/pathfinder/pathfinder $(BINDIR)/$(BINSUBDIR)/pathfinder-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/lavaMD/lavaMD $(BINDIR)/$(BINSUBDIR)/lavaMD-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/lud/cuda/lud_cuda $(BINDIR)/$(BINSUBDIR)/lud-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/leukocyte/CUDA/leukocyte $(BINDIR)/$(BINSUBDIR)/leukocyte-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/hotspot3D/3D $(BINDIR)/$(BINSUBDIR)/hotspot3D-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/gaussian/gaussian $(BINDIR)/$(BINSUBDIR)/gaussian-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/srad/srad_v2/srad $(BINDIR)/$(BINSUBDIR)/srad_v2-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/srad/srad_v1/srad $(BINDIR)/$(BINSUBDIR)/srad_v1-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/backprop $(BINDIR)/$(BINSUBDIR)/backprop-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/bfs  $(BINDIR)/$(BINSUBDIR)/bfs-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/euler3d $(BINDIR)/$(BINSUBDIR)/cfd-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/hotspot $(BINDIR)/$(BINSUBDIR)/hotspot-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/kmeans $(BINDIR)/$(BINSUBDIR)/kmeans-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/needle $(BINDIR)/$(BINSUBDIR)/needle-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/streamcluster $(BINDIR)/$(BINSUBDIR)/streamcluster-rodinia-3.1
	mv $(BINDIR)/mummergpu $(BINDIR)/$(BINSUBDIR)/mummergpu-rodinia-3.1

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


