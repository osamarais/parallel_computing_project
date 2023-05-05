all: main.c ga_gpu.cu
	mpicc -g main.c -c -o main.o
	nvcc -g -G -arch=sm_70 ga_gpu.cu -c -o ga-cuda.o
	mpicc -g main.o ga-cuda.o -o ga-cuda-mpi-exe -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
