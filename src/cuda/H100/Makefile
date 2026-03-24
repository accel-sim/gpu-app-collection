# H100 Benchmark Suite Makefile (following cutlass-bench pattern)
.PHONY: all simple graph newton vpi clean cufft cusolver image

# Default: Build all apps (submodules initialized by parent Makefile)
all: simple graph newton vpi

# Simple apps (cuFFT, cuSolver, image processing - CUDA toolkit only)
simple: cufft cusolver image

cufft:
	$(MAKE) -C cufft/cufft_3d_c2c
	$(MAKE) -C cufft/cufft_lto_r2c_c2r

cusolver:
	$(MAKE) -C cusolver/cusolver_ormqr
	$(MAKE) -C cusolver/cusolver_Xgetrf

image:
	$(MAKE) -C image/dwtHaar1D
	$(MAKE) -C image/recursiveGaussian
	$(MAKE) -C image/FDTD3d

# Graph apps (cuGraph submodule - built with CMake)
graph:
	@echo "Building cuGraph library from submodule..."
	@if [ -d "external/cugraph" ]; then \
		cd external/cugraph && \
		./build.sh libcugraph --skip_cpp_tests && \
		echo "Building graph apps..." && \
		cd ../.. && \
		export CUGRAPH_ROOT="$(shell pwd)/external/cugraph" && \
		mkdir -p graph/bfs_standalone/build && \
		cd graph/bfs_standalone/build && \
		cmake .. -DCMAKE_CUDA_COMPILER=$(CUDA_INSTALL_PATH)/bin/nvcc && \
		$(MAKE) && \
		cd ../../.. && \
		mkdir -p graph/mst_standalone/build && \
		cd graph/mst_standalone/build && \
		cmake .. -DCMAKE_CUDA_COMPILER=$(CUDA_INSTALL_PATH)/bin/nvcc && \
		$(MAKE); \
	else \
		echo "WARNING: cuGraph submodule not found - skipping graph apps"; \
	fi

# Newton apps (Newton submodule - Python-based)
newton:
	@echo "Newton apps ready (Python-based, no build needed)"

# VPI apps (VPI library from system - required)
vpi:
	@echo "Searching for VPI installation..."
	@VPI_ROOT=$$(find /opt/nvidia -maxdepth 1 -name "vpi*" -type d 2>/dev/null | head -1); \
	if [ -z "$$VPI_ROOT" ]; then \
		echo "ERROR: VPI not found in /opt/nvidia/"; \
		echo "Please install VPI from NVIDIA"; \
		exit 1; \
	fi; \
	echo "Found VPI at $$VPI_ROOT"; \
	export VPI_ROOT=$$VPI_ROOT; \
	mkdir -p vpi/vpi_background_subtractor/build && \
	cd vpi/vpi_background_subtractor/build && \
	cmake .. -DVPI_ROOT=$$VPI_ROOT && $(MAKE) && \
	cd ../../.. && \
	mkdir -p vpi/vpi_orb_feature_detector/build && \
	cd vpi/vpi_orb_feature_detector/build && \
	cmake .. -DVPI_ROOT=$$VPI_ROOT && $(MAKE) && \
	cd ../../.. && \
	mkdir -p vpi/vpi_stereo_disparity/build && \
	cd vpi/vpi_stereo_disparity/build && \
	cmake .. -DVPI_ROOT=$$VPI_ROOT && $(MAKE)

clean:
	# Clean bin directory
	-rm -rf bin
	# Clean simple apps
	-$(MAKE) -C cufft/cufft_3d_c2c clean
	-$(MAKE) -C cufft/cufft_lto_r2c_c2r clean
	-$(MAKE) -C cusolver/cusolver_ormqr clean
	-$(MAKE) -C cusolver/cusolver_Xgetrf clean
	-$(MAKE) -C image/dwtHaar1D clean
	-$(MAKE) -C image/recursiveGaussian clean
	-$(MAKE) -C image/FDTD3d clean
	# Clean graph apps and cuGraph build
	-rm -rf graph/bfs_standalone/build
	-rm -rf graph/mst_standalone/build
	-rm -rf external/cugraph/cpp/build
	# Clean Newton
	-rm -rf newton/newton_venv
	# Clean VPI apps
	-rm -rf vpi/*/build
