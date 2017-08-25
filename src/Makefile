
ifeq ($(GPGPUSIM_BENCHMARKS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

all:   pannotia rodinia_2.0-ft proxy-apps dragon-naive 
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
# For Dragon, we need to change the archs manually!  (TO DO)
# Naive works wwith only SM_20 and above
# Fro Dragon-cdp comilation, you need to ensure that you are using at least CUDA 5.5
###################################################################################################3
dragon-naive: data
	chmod +x cuda/dragon_li/sconstruct
	scons sm20=1 -C cuda/dragon_li

dragon-cdp: data
	chmod +x cuda/dragon_li/sconstruct
	scons cdp=1 -C cuda/dragon_li

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
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/minife_matvec_ell
	$(SETENV) make $(MAKE_ARGS) -C cuda/proxy-apps-doe/xsbench

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
