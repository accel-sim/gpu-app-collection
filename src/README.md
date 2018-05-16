### Dependencies

This repo is meant to be as self-contained as possible.
However:
cuda, gcc, make and scons
are required to build all the apps.
To build the bechmarks:

```bash
export CUDA_INSTALL_PATH=<whatever yours is the build scripts should be robust enough to build with any CUDA version. Versions 4.2 and 9.1 are extensively tested>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
source setup_environment
make -i -j all
make data
```

To see how many apps built:
```bash
ls ../bin/9.1/release/ | wc
```
