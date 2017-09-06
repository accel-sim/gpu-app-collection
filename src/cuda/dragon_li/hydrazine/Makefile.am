################################################################################
#	\file Makefile.am
#	\author Gregory Diamos
#	\date Saturday May 16, 2009
# 	\brief Automake input file describing the programs and sources needed to
#		Hydrazine
################################################################################

################################################################################
## Programs and Libraries
bin_PROGRAMS = 
check_PROGRAMS = TestActiveTimer TestArgumentParser TestCudaVector \
	TestMath \
	TestThread TestTimer TestXmlArgumentParser \
	TestXmlParser TestBTree TestJson
lib_LIBRARIES = libhydralize.a
################################################################################

################################################################################
## libhydralize
libhydralize_a_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
libhydralize_a_SOURCES = \
	hydrazine/implementation/Exception.cpp \
	hydrazine/implementation/XmlLexer.cpp \
	hydrazine/implementation/Timer.cpp \
	hydrazine/implementation/LowLevelTimer.cpp \
	hydrazine/implementation/ArgumentParser.cpp \
	hydrazine/implementation/XmlParser.cpp \
	hydrazine/implementation/math.cpp \
	hydrazine/implementation/debug.cpp \
	hydrazine/implementation/XmlArgumentParser.cpp \
	hydrazine/implementation/XmlTree.cpp \
	hydrazine/implementation/string.cpp \
	hydrazine/implementation/json.cpp \
	hydrazine/implementation/Configurable.cpp \
	hydrazine/implementation/Test.cpp \
	hydrazine/implementation/ActiveTimer.cpp \
	hydrazine/implementation/Thread.cpp \
	hydrazine/implementation/Version.cpp \
	hydrazine/implementation/SystemCompatibility.cpp
################################################################################

################################################################################
## TestActiveTimer
TestActiveTimer_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestActiveTimer_SOURCES = hydrazine/test/TestActiveTimer.cpp
TestActiveTimer_LDADD = libhydralize.a
TestActiveTimer_LDFLAGS =
################################################################################

################################################################################
## TestArgumentParser
TestArgumentParser_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestArgumentParser_SOURCES = hydrazine/test/TestArgumentParser.cpp
TestArgumentParser_LDADD = libhydralize.a
TestArgumentParser_LDFLAGS =
################################################################################

################################################################################
## TestBTree
TestBTree_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestBTree_SOURCES = hydrazine/test/TestBTree.cpp
TestBTree_LDADD = libhydralize.a
TestBTree_LDFLAGS =
################################################################################

################################################################################
## TestCudaVector
TestCudaVector_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestCudaVector_SOURCES = hydrazine/test/TestCudaVector.cpp
TestCudaVector_LDADD = libhydralize.a
TestCudaVector_LDFLAGS = $(CUDA_LIBS)
################################################################################

################################################################################
## TestMath
TestMath_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestMath_SOURCES = hydrazine/test/TestMath.cpp
TestMath_LDADD = libhydralize.a
TestMath_LDFLAGS =
################################################################################

################################################################################
## TestThread
TestThread_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestThread_SOURCES = hydrazine/test/TestThread.cpp
TestThread_LDADD = libhydralize.a
TestThread_LDFLAGS =
################################################################################

################################################################################
## TestTimer
TestTimer_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestTimer_SOURCES = hydrazine/test/TestTimer.cpp
TestTimer_LDADD = libhydralize.a
TestTimer_LDFLAGS =
################################################################################

################################################################################
## TestXmlArgumentParser
TestXmlArgumentParser_CXXFLAGS = -Wall -ansi -pedantic -Werror \
	-std=c++0x
TestXmlArgumentParser_SOURCES = hydrazine/test/TestXmlArgumentParser.cpp
TestXmlArgumentParser_LDADD = libhydralize.a
TestXmlArgumentParser_LDFLAGS =
################################################################################

################################################################################
## TestXmlParser
TestXmlParser_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestXmlParser_SOURCES = hydrazine/test/TestXmlParser.cpp
TestXmlParser_LDADD = libhydralize.a
TestXmlParser_LDFLAGS =
################################################################################

################################################################################
## TestJson
TestJson_CXXFLAGS = -Wall -ansi -pedantic -Werror -std=c++0x
TestJson_SOURCES = hydrazine/test/TestJson.cpp
TestJson_LDADD = libhydralize.a
TestJson_LDFLAGS =
################################################################################

################################################################################
## Tests
regression/regression.log : check
	@echo
	@echo "Running Hydrazine Regression Tests"
	@python hydrazine/python/RunRegression.py \
		-t regression/tests.txt \
		-l regression/regression.log -v

test : regression/regression.log
################################################################################

