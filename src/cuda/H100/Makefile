# H100 Benchmark Suite Makefile (following cutlass-bench pattern)
.PHONY: all simple graph newton vpi clean

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
		./build.sh libcugraph --skip_cpp_tests -n && \
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
	@echo "Setting up Newton apps..."
	chmod +x newton/setup_newton.sh newton/newton_diffsim_ball newton/newton_robot_cartpole newton/newton_mpm_granular
	@echo "Newton wrapper scripts ready. Virtual environment will be created on first run."

# VPI apps (VPI library from system - optional)
# NOTE: VPI requires system installation. Run setup_vpi.sh with sudo if VPI is not installed.
vpi:
	@if command -v vpi-config &> /dev/null; then \
		echo "Building VPI apps..."; \
		mkdir -p vpi/vpi_background_subtractor/build && \
		cd vpi/vpi_background_subtractor/build && \
		cmake .. && $(MAKE) && \
		cd ../../.. && \
		mkdir -p vpi/vpi_orb_feature_detector/build && \
		cd vpi/vpi_orb_feature_detector/build && \
		cmake .. && $(MAKE) && \
		cd ../../.. && \
		mkdir -p vpi/vpi_stereo_disparity/build && \
		cd vpi/vpi_stereo_disparity/build && \
		cmake .. && $(MAKE); \
	else \
		echo "WARNING: VPI not installed - skipping VPI apps"; \
		echo "  To install VPI, run: sudo bash setup_vpi.sh"; \
	fi

clean:
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
