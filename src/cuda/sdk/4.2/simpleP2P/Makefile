################################################################################
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
# 
#
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= simpleP2P
# Cuda source files (compiled with cudacc)
CUFILES		:= simpleP2P.cu

# Do not link with CUTIL and SHRUTIL
OMIT_CUTIL_LIB   := 1
OMIT_SHRUTIL_LIB := 1

################################################################################
# Rules and targets

include ../../common/common.mk

################################################################################
# Additional libraries to include

LIB += -lpthread
