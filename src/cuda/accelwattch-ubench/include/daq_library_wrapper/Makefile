SHELL = /bin/sh

LIBS = nidaqmx
LIBFLAGS = -l$(LIBS)
TARGET = libourdaq.so
CC = g++


CDEBUG = -g

CFLAGS += $(CDEBUG)

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) core

#ContAcq-IntClk.o : ContAcq-IntClk.c
#	$(CC) -c -fPIC ContAcq-IntClk.c -o ContAcq-IntClk.o

$(TARGET) : ContAcq-IntClk.c
	$(CC) -shared -fPIC -lnidaqmx -o libourdaq.so ContAcq-IntClk.c

