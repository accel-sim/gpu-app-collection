#ifndef LSG_KERNELCONFIG
#define LSG_KERNELCONFIG

typedef struct KernelConfig {
	unsigned device;
	unsigned problemsize;
	unsigned nblocks, blocksize;
	cudaDeviceProp dp;

	KernelConfig(unsigned ldevice = 0);
	void	 init();
	unsigned setProblemSize(unsigned size);
	unsigned setNumberOfBlocks(unsigned lnblocks);
	unsigned setNumberOfBlockThreads(unsigned lblocksize);
	unsigned setMaxThreadsPerBlock();
	unsigned getNumberOfBlocks();
	unsigned getNumberOfBlockThreads();
	unsigned getNumberOfTotalThreads();

	unsigned calculate();
	unsigned getMaxThreadsPerBlock();
	unsigned getMaxBlocks();
	unsigned getMaxSharedMemoryPerBlock();
	unsigned getNumberOfSMs();
	bool	 coversProblem(unsigned size = 0);
	unsigned getProblemSize();
} KernelConfig;

KernelConfig::KernelConfig(unsigned ldevice/* = 0*/) {
	device = ldevice;
	init();
}
void KernelConfig::init() {
	int deviceCount = 0;
	if (cudaSuccess != cudaGetDeviceCount(&deviceCount)) {
		CudaTest("cudaGetDeviceCount failed");
	}
	if (deviceCount == 0) {
        	fprintf(stderr, "No CUDA capable devices found.");
		return;
	} 

	cudaGetDeviceProperties(&dp, device);
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", deviceCount, device, dp.name, dp.major, dp.minor, getNumberOfSMs(), ConvertSMVer2Cores(dp.major, dp.minor));
	problemsize = 0;
	nblocks = 0;
	setMaxThreadsPerBlock();	// default.
}
unsigned KernelConfig::getMaxThreadsPerBlock() {
	return dp.maxThreadsDim[0];
}
unsigned KernelConfig::getMaxBlocks() {
	return dp.maxGridSize[0];
}
unsigned KernelConfig::getMaxSharedMemoryPerBlock() {
	return dp.sharedMemPerBlock;
}
unsigned KernelConfig::getNumberOfSMs() {
	return dp.multiProcessorCount;
}

unsigned KernelConfig::setProblemSize(unsigned size) {
	problemsize = size;
	return calculate();
}
unsigned KernelConfig::getProblemSize() {
	return problemsize;
}
unsigned KernelConfig::getNumberOfBlocks() {
	return nblocks;
}
unsigned KernelConfig::getNumberOfBlockThreads() {
	return blocksize;
}
unsigned KernelConfig::getNumberOfTotalThreads() {
	return nblocks * blocksize;
}
unsigned KernelConfig::calculate() {
	if (blocksize == 0) {
		fprintf(stderr, "blocksize = 0.\n");
		return 1;
	}
	nblocks = (problemsize + blocksize - 1) / blocksize;
	return 0;
}
unsigned KernelConfig::setNumberOfBlocks(unsigned lnblocks) {
	nblocks = lnblocks;
	return nblocks;
}
unsigned KernelConfig::setNumberOfBlockThreads(unsigned lblocksize) {
	blocksize = lblocksize;
	return blocksize;
}
unsigned KernelConfig::setMaxThreadsPerBlock() {
	return setNumberOfBlockThreads(getMaxThreadsPerBlock());
}
bool KernelConfig::coversProblem(unsigned size/* = 0*/) {
	if (size == 0) {
		size = problemsize;
	}
	return (size <= nblocks * blocksize);
}
#endif
