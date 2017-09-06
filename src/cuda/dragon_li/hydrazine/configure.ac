################################################################################
#	\file configure.ac
#	\author Gregory Diamos <gregory.diamos@gatech.edu>
#	\date Saturday May 16, 2009
# 	\brief Autoconf input file describing the checks that need to be performed 
#			to create a configure script for the hydrazine library.
################################################################################

################################################################################
## Package specification
# Revision scheme is MajorFeatureSet.MinorFeatureSet.SVNChangelistNumber
AC_PREREQ(2.61)
AC_INIT(hydrazine, 0.1.2, gregory.diamos@gatech.edu)
AM_INIT_AUTOMAKE([])
################################################################################

################################################################################
## Package Configuration
AC_CONFIG_HEADER([configure.h])
AC_CONFIG_MACRO_DIR([m4])
################################################################################

################################################################################
## Checks for libraries.
AC_CHECK_LIB([rt], [clock_gettime], [],\
	AC_MSG_ERROR("librt required"))
################################################################################

################################################################################
## Checks for programs.
AC_PROG_CXX
AC_PROG_CC
LT_INIT
################################################################################

################################################################################
# Checks for header files.
AC_CHECK_HEADERS([float.h])
################################################################################

################################################################################
# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])
################################################################################

################################################################################
# Checks for library functions.
AC_CHECK_FUNCS([clock_gettime getpagesize memmove memset pow])
################################################################################

###############################################################################
# Setup CUDA paths
# 
AC_ARG_WITH([cuda],
	[--with-cuda=PATH prefix where cuda is installed [default=/usr/local/cuda]])

if test -n "$with_cuda"
then
	CUDA_CFLAGS="-I$with_cuda/include"
	CUDA_LIBS="-L$with_cuda/lib64 -lcudart -lcublas"
	CUFFT_LIBS="-L$with_cuda/lib64 -lcufft"
	CUBLAS_LIBS="-L$with_cuda/lib64 -lcublas"
	NVCC="$with_cuda/bin/nvcc"
else
	CUDA_CFLAGS="-I/usr/local/cuda/include"
	CUDA_LIBS="-L/usr/local/cuda/lib64 -lcudart -lcublas"
	CUFFT_LIBS="-L/usr/local/cuda/lib64 -lcufft"
	CUBLAS_LIBS="-L/usr/local/cuda/lib64 -lcublas"
	NVCC="nvcc"
fi

AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUFFT_LIBS)
AC_SUBST(CUBLAS_LIBS)
AC_SUBST(NVCC)
################################################################################

################################################################################
## Output generation
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
################################################################################

