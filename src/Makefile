
ifeq ($(GPGPUSIM_BENCHMARKS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

ifeq ($(CUDA_VERSION_MAJOR), 4)
all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive microbench sdk-4.2 rodinia-3.1
else
all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive microbench rodinia-3.1
endif
#  dragon-cdp 

#Disable clean for now, It has a bug!
#clean: clean_rodinia_2.0-ft 
#	../data_dirs/clean_data.sh

#clean_dragon-naive clean_dragon-cdp clean_pannotia clean_proxy-apps

data:
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
	$(SETENV) make $(MAKE_ARGS) noinline=$(noinline) -C cuda/sdk/4.2/$@

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

clean_$(RODINIA_31_APPS):
	if [ -f cuda/rodinia/3.1/cuda/$@/Makefile_nvidia ]; then \
		$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@ -f Makefile_nvidia; \
	elif [ -f cuda/rodinia/3.1/cuda/$@/CUDA/Makefile ]; then \
		$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@/CUDA; \
	else \
		$(SETENV) make $(MAKE_ARGS) clean noinline=$(noinline) -C cuda/rodinia/3.1/cuda/$@; \
	fi

rodinia-3.1: $(RODINIA_31_APPS)
	mv cuda/rodinia/3.1/cuda/b+tree/b+tree.out $(BINDIR)/$(BINSUBDIR)/b+tree.out-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/dwt2d/dwt2d $(BINDIR)/$(BINSUBDIR)/dwt2d-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/heartwall/heartwall $(BINDIR)/$(BINSUBDIR)/heartwall-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/huffman/pavle $(BINDIR)/$(BINSUBDIR)/huffman-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/hybridsort/hybridsort $(BINDIR)/$(BINSUBDIR)/hybridsort-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/myocyte/myocyte.out $(BINDIR)/$(BINSUBDIR)/myocyte.out-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/nn/nn $(BINDIR)/$(BINSUBDIR)/nn-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/particlefilter/particlefilter_float $(BINDIR)/$(BINSUBDIR)/particlefilter_float-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/particlefilter/particlefilter_naive $(BINDIR)/$(BINSUBDIR)/particlefilter_naive-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/pathfinder/pathfinder $(BINDIR)/$(BINSUBDIR)/pathfinder-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/lavaMD/a.out $(BINDIR)/$(BINSUBDIR)/lavaMD-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/lud/cuda/lud_cuda $(BINDIR)/$(BINSUBDIR)/lud-rodinia-3.1
	mv cuda/rodinia/3.1/cuda/leukocyte/CUDA/leukocyte $(BINDIR)/$(BINSUBDIR)/leukocyte-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/backprop $(BINDIR)/$(BINSUBDIR)/backprop-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/bfs  $(BINDIR)/$(BINSUBDIR)/bfs-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/euler3d $(BINDIR)/$(BINSUBDIR)/euler3d-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/hotspot $(BINDIR)/$(BINSUBDIR)/hotspot-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/kmeans $(BINDIR)/$(BINSUBDIR)/kmeans-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/needle $(BINDIR)/$(BINSUBDIR)/needle-rodinia-3.1
	mv $(BINDIR)/$(BINSUBDIR)/streamcluster $(BINDIR)/$(BINSUBDIR)/streamcluster-rodinia-3.1
	mv $(BINDIR)/mummergpu $(BINDIR)/$(BINSUBDIR)/mummergpu-rodinia-3.1

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


