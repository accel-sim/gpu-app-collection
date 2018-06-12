all:
	nvcc -O3 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe