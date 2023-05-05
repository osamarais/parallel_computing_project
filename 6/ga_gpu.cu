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
// Fitness
extern double *fitness;
// Global roulette indices
extern unsigned long long *globalRouletteWheel;

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



extern "C" void fitnessAlloc(int num_ranks, int rank, int popSize){

  // Allocate the memory
  int localSize = popSize/num_ranks;
  cudaMallocManaged(&fitness, localSize*(sizeof(double)));
  



}





// Fitness calculation Kernel
__global__ void fitness_kernel(int num_ranks, int rank, int popSize, int cities, unsigned char *currentGen, double *fitness, double *map){
  
  int localPopSize = popSize/num_ranks;
 
 
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (;
       index < localPopSize;
       index += blockDim.x * gridDim.x){

    int j = 0;
    double sum = 0;

    int a = 0;
    int b = 0;
    double distance=0;

    for(j=0;j<cities-1;j++){
      
      
      a = currentGen[rank*cities*localPopSize + index*cities + j];
      b = currentGen[rank*cities*localPopSize + index*cities + j+1];

      distance = map[a*cities + b];

      sum = sum + distance;
    }

    // add the last two
    a = currentGen[rank*cities*localPopSize + index*cities];
    b = currentGen[rank*cities*localPopSize + index*cities + cities-1];
    distance = map[a*cities + b];
    sum = sum + distance;

    fitness[index] = sum;

  }

  

}





// Function to launch the Fitness Kernel
extern "C" bool fitness_kernelLaunch (int num_ranks, int rank, int popSize, int cities)
{
  // get the minimum number of required blockCount
  //size_t blockCount = (worldWidth * worldHeight + threadsCount - 1)/threadsCount;
  size_t threadsCount = 1024;
  size_t blockCount = popSize/threadsCount+1;
  if(blockCount == 0) blockCount++;
  //printf("blockCount: %d \n", blockCount);
  fitness_kernel<<<blockCount, threadsCount>>>(num_ranks, rank, popSize, cities, currentGen, fitness, map);
  cudaDeviceSynchronize();

  return false;
}

















__global__ void parents_kernel(int num_ranks, int popSize, int cities, unsigned char *currentGen, unsigned char *nextGen, int r_wheel_length, unsigned long long *globalRouletteWheel){
  
  int localPopSize = popSize/num_ranks;
 
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  int nextgen_i = index + (index/r_wheel_length)*(localPopSize - r_wheel_length);
  int currentgen_i = globalRouletteWheel[index];

  int j = 0;
  for (;
       index < r_wheel_length*num_ranks;
       index += blockDim.x * gridDim.x){

    for(j=0;j<cities;j++){
      nextGen[nextgen_i*cities + j] = currentGen[currentgen_i*cities +j];
    }


  }
}
 
// Function to launch the Fitness Kernel
extern "C" bool parents_kernelLaunch (int num_ranks, int r_wheel_length)
{
  // get the minimum number of required blockCount
  //size_t blockCount = (worldWidth * worldHeight + threadsCount - 1)/threadsCount;
  size_t threadsCount = 1024;
  size_t blockCount = r_wheel_length/threadsCount+1;
  if(blockCount == 0) blockCount++;
  //printf("blockCount: %d \n", blockCount);
  parents_kernel<<<blockCount, threadsCount>>>(num_ranks, popSize, cities, currentGen, nextGen, r_wheel_length, globalRouletteWheel);
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






