// Genetic Algorithm on Multiple GPUs


// ga_gpu.cu contains all the CUDA routines




#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>


// Extern variables to make them available here in the cuda file

// Result from last compute of world.
extern unsigned char *currentGen;
// Current state of world. 
extern unsigned char *nextGen;
// Map
extern double *map;


// Problem Sizes
// Population Size
// Cities/Genome Length
extern unsigned long long popSize;
extern unsigned long long cities;
extern unsigned long long globalSize;



// Bind GPUs to the ranks
extern "C" void bindGPUs(int rank)
{

  // Bind GPUs to the ranks
  int cudaDeviceCount;
  int cE;

  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
  {
    printf(" Unable to determine cuda device count, error is %d, count is %d\n",
      cE, cudaDeviceCount );
    exit(-1);
  }

  if( (cE = cudaSetDevice( rank % cudaDeviceCount )) != cudaSuccess )
  {
    printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
      rank, (rank % cudaDeviceCount), cE);
    exit(-1);
  }



}



// Initialization Routine
extern "C" void popAlloc(int cities, int popSize){

  // Allocate the memory
  globalSize = cities*popSize;
  cudaMallocManaged(&currentGen, globalSize*(sizeof(unsigned char)));
  cudaMallocManaged(&nextGen, globalSize*(sizeof(unsigned char)));



}


extern "C" void mapAlloc(unsigned long long cities){
// cuda malloc manage the parsed map ----> copy it into variable called map
  unsigned long long mapsize = cities*cities;
  cudaMallocManaged(&map, mapsize*(sizeof(double)));
}




// Fitness calculation Kernel
__global__ void fitness_kernel(){
// 
}





// Function to launch the Fitness Kernel
extern "C" bool fitness_kernelLaunch ()
{
  // get the minimum number of required blockCount
  //size_t blockCount = (worldWidth * worldHeight + threadsCount - 1)/threadsCount;
  size_t threadsCount = 1;
  size_t blockCount = 1;
  if(blockCount == 0) blockCount++;
  //printf("blockCount: %d \n", blockCount);
  fitness_kernel<<<blockCount, threadsCount>>>();
  cudaDeviceSynchronize();

  return false;
}

 






// Function to launch the cum sum kernel
// Kernel for cum sum (scan)
void cum_kernelLaunch(){
}
//

// Function to launch the copying for next generation parents
// Kernel for copying


// Function to launch the crossover kernel
// crossover kernel






