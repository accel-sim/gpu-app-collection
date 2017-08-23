matvec: *.cu *.h
	nvcc main.cu -g -Xptxas -dlcm=cg -arch=sm_20 -o matvec -Xptxas -v

clean: 
	rm -f matvec
