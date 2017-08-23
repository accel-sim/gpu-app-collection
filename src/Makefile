
ifeq ($(GPGPUSIM_BENCHMARKS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

all:  rodinia_2.0-ft proxy-apps dragon-naive dragon-cdp pannotia

clean: clean_rodinia_2.0-ft clean_dragon-naive clean_dragon-cdp clean_pannotia clean_proxy-apps
	../data_dirs/clean_data.sh

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

dragon-naive: data
	chmod +x cuda/dragon_li/sconstruct
	scons -C cuda/dragon_li

dragon-cdp: data
	chmod +x cuda/dragon_li/sconstruct
	scons cdp=1 -C cuda/dragon_li

pannotia: data
	$(SETENV) make -C cuda/pannotia/bc
	$(SETENV) export VARIANT="MAX"; make -C cuda/pannotia/color
	$(SETENV) export VARIANT="MAXMIN"; make -C cuda/pannotia/color
	$(SETENV) export VARIANT="DEFAULT"; make -C cuda/pannotia/fw
	$(SETENV) export VARIANT="BLOCK"; make -C cuda/pannotia/fw
	$(SETENV) make -C cuda/pannotia/mis
	$(SETENV) export VARIANT="DEFAULT"; make -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="SPMV"; make -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="DEFAULT"; make -C cuda/pannotia/sssp
	$(SETENV) export VARIANT="ELL"; make -C cuda/pannotia/sssp

proxy-apps: data
	#chmod +x cuda/proxy-apps-doe/cns/compile.bash
	#cd cuda/proxy-apps-doe/cns
	#exec ./cuda/proxy-apps-doe/cns/compile.bash -C cuda/proxy-apps-doe/cns
	#chmod +x cuda/proxy-apps-doe/comd/cmd_compile.bash
	#./cuda/proxy-apps-doe/comd/cmd_compile.bash 
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
	$(SETENV) make clean -C cuda/pannotia/bc
	$(SETENV) export VARIANT="MAX"; make clean -C cuda/pannotia/color
	$(SETENV) export VARIANT="MAXMIN"; make clean -C cuda/pannotia/color
	$(SETENV) export VARIANT="DEFAULT"; make clean -C cuda/pannotia/fw
	$(SETENV) export VARIANT="BLOCK"; make clean -C cuda/pannotia/fw
	$(SETENV) make -C cuda/pannotia/mis
	$(SETENV) export VARIANT="DEFAULT"; make clean -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="SPMV"; make clean -C cuda/pannotia/pagerank
	$(SETENV) export VARIANT="DEFAULT"; make clean -C cuda/pannotia/sssp
	$(SETENV) export VARIANT="ELL"; make clean -C cuda/pannotia/sssp

clean_proxy-apps:

