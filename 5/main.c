// Genetic Algorithm on Multiple GPUs





// Include all the relevant headers for MPI and running main
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<time.h>
#include<string.h>
#include<math.h>
#include<ctype.h>

#define CITY_MAX 2000


// Initialize and Declare Current and Next populations
// Result from last compute of world.
unsigned char *currentGen=NULL;
// Current state of world. 
unsigned char *nextGen=NULL;
// Map has to be on the GPU
double *map;
// Fitness array on the GPU
double *fitness;
// Cumulative fitness array, local
double *localCum;
// Cumulative Sum Array, responsilbe for the entire population
double *cumFitness;
// Offsets: for the cumulative sum
double *offsets;
// Roulette Indices
unsigned long long *rouletteWheel;
// Global Roulette that will be synchronized
unsigned long long *globalRouletteWheel;
// Parents Index List
unsigned long long *parentsList;




// Problem Sizes
// Population Size
unsigned long long popSize=0;
// Cities/Genome Length
unsigned long long cities=0;
// Total Size of 1D array
unsigned long long globalSize=0;











// Prototypes of externed functions from the CUDA file
void bindGPUs(int rank);
void popAlloc(int cities, int popSize);
void mapAlloc(unsigned long long cities);
void fitnessAlloc(int num_ranks, int rank, int popSize);
//bool fitness_kernelLaunch();


// Protype of functions
double** distance_cities(double*, int);
int file_read(char* file_name);

void popInit(int num_ranks, int rank, int cities, int popSize);
void syncPop(int num_ranks, int rank);

bool fitness_kernelLaunch (int num_ranks, int rank, int popSize, int cities);

void calculateCum(int popSize, int num_ranks);

void syncCum(int num_ranks, int rank);

void populateOffsets(int num_ranks, int popSize);

void modifyCumulative(int num_ranks, int popSize);

void rouletteIndices(int r_wheel_length, int rank);

void syncRoulette(int num_ranks,int r_wheel_length);



int main(int argc, char** argv) {


// Read in argc and argv
// Parse the argc
  if( argc != 3 )
  {
    printf("GA for TSP requires 2 argument: Matrix of Cities and the output flag, e.g. ./main cities.txt 0 \n");
    exit(-1);
  }

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Get the number of processes
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Print off a hello world message
  printf("Hello world from rank %d out of %d processors\n", rank, num_ranks);
  // Record the MPI time
  double t1;
  if(0==rank)
  {
    t1 = MPI_Wtime();
  }

  // Parse argc argv
  char *filename;
  int save_flag;
  filename = argv[1];
  save_flag = atoi(argv[2]);

  // Now bind gpus to ranks
  bindGPUs(rank);
  

  double crossoverRatio=0.8;
  double mutationRatio=0.05;

  // Read in the file, get the number of cities
  cities =  file_read(filename);

  // Initialize cities^2 population size
  popSize = ((cities*cities)/num_ranks)*num_ranks;
  // Now we want to set the GPU device for each rank, and initialize the worlds
  popAlloc(cities, popSize);
  // generate the rank's portion of the initial population
  // do this on the CPU
  popInit(num_ranks, rank, cities, popSize);
  

  // Round down this value
  int r_wheel_length = floor(crossoverRatio*popSize/num_ranks);
  int local_children_num = popSize/num_ranks - r_wheel_length;  





  // Cop the map array held on the GPU
  //mapInit(map,cities,parsedMap);
  // Allocate the rouletteWheel array
  //rouletteAlloc(const crossoverRatio,const &popSize, &rouletteWheel)
  // Allocate the Parents List array
  //parentsAlloc(const crossoverRatio, const &popSize, &parentsList)

//////////////////////////////////////////////////////////////////
// Share the data using MPI to other ranks to synchronize
// Function to synchronize population across ranks
  syncPop(num_ranks, rank);

  
  MPI_Barrier(MPI_COMM_WORLD);
  int i = 0;
  int j = 0;
  if(0==rank){
    for( i = 0 ; i<popSize ; i++ ){
      printf("Individual %d: ",i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",currentGen[cities*i + j]);
      }
      printf("\n");
    }
  }

  printf("Allocating fitness array on GPU\n");
  localCum = calloc(popSize/num_ranks, sizeof(double));
  cumFitness = calloc(popSize, sizeof(double));
  offsets = calloc(num_ranks, sizeof(double));
  globalRouletteWheel = calloc(r_wheel_length*num_ranks,sizeof(unsigned long long));
  


// Run loop till converged
  //while (0){

// Calculate Fitness Values
  // Allocate fitness array
  fitnessAlloc(num_ranks, rank, popSize);
  //printf("Allocated fitness array on GPU\n");
  //printf("Launching fitness kernel\n");
  fitness_kernelLaunch(num_ranks, rank, popSize, cities);
  //printf("Launched fitnedd kernel\n");

  for(i=0;i<cities;i++){
    for(j=0;j<cities;j++){
      //printf("%f ",map[i*cities +j]);
    }
    //printf("\n");
  }
  
  //printf("Printing fitness array\n");
  
  for(i = 0; i < popSize/num_ranks; i++){
    //printf("Fitness of Individual %llu : %f\n",rank*popSize/num_ranks+i,fitness[i]);
  }


  //printf("Calculating the cumulative fitness \n");
  // Calculate the cumulative fitness on the CPU; will be faster on the CPU
  calculateCum(popSize, num_ranks);

  //printf("Printing the local Cumulative sum\n");
  
  for(i = 0; i < popSize/num_ranks; i++){
    //printf("Cumulative sum at Individual %llu : %f\n",rank*popSize/num_ranks+i,localCum[i]);
  }

  
  //printf("Synchronizing the cumulative sums\n");
  // Synchronize the cumulative fitness arrays localCum to cumFitness
  syncCum(num_ranks,rank);
  if (0==rank){
    printf("Printing the cumulative sums\n");
    for(i = 0; i < popSize; i++){
      printf("Individual %d : %f\n", i,cumFitness[i]);
    }
  }


  printf("Populating offsets\n");
  populateOffsets(num_ranks, popSize);
  if (0==rank){
    printf("Printing the offsets\n");
    for(i = 0; i < num_ranks; i++){
      printf("Rank %d : %f\n", i,offsets[i]);
    }
  }
  
  modifyCumulative(num_ranks, popSize);
  if (0==rank){
    printf("Printing modified cumulative sums\n");
    for(i = 0; i < popSize; i++){
      printf("Individual %d : %f\n", i,cumFitness[i]);
    }
  }  


  printf("Generating the Roulette Wheel\n");
  // Generate the roulette wheel for this rank
  rouletteIndices(r_wheel_length, rank);

  if(0==rank){
    printf("Next gen individuals after roulette");
    for( i = 0 ; i<popSize ; i++ ){
      printf("Next Individual %d: ",i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",nextGen[cities*i + j]);
      }
      printf("\n");
    }
  }

  syncRoulette(num_ranks,r_wheel_length);

  if(0==rank){
    printf("Printing the global roulette wheel");
    for(i=0;i<num_ranks*r_wheel_length; i++){
      printf("%llu \n",globalRouletteWheel[i]);
    }
  }

  
  
  


// Save the best one if better than previous ones
// Do this when neeeded
// Exit loop if converged

// Calculate a cumulative sum on its own fitness array
// Launch cum sum kernel
    //cum_kernelLaunch(const &fitness, &cumFitness, rank)
// Share the cum sum offsets using MPI
    //syncCum(&cumFitness, rank)
// Pick out the last elements of the cumulative sum, put them in offsets
    //populateOffsets(const &cumFitness, &offset)

// Roulette Selection
// Generate an array of indices
// use sizeof to find array sizes
    //rouletteIndices(const &offsets, const &cumFitness, &rouletteWheel);

////////// Share indices across the ranks to reduce MPI overhead ///////////

// Perform the copying using a kernel
    //launchCopy(const &rouletteWheel,const &currentGen, &nextGen);


// Crossover
// Populate parents list
    //popParents(const popSize, &parentsList)
// Generate the random array for parent pairs for crossover
// Launch crossover kernel
    //launchCrossover(const currentGen, nextGen, &rank, const &popParents);


// sync the crossover children. The parents were moved using indices earlier
    //syncChildren(&nextGen);



// Mutation
// Only on its own rank
    // This is a CPU function
    //mutate();

// Send Mutations to the other ranks if they have occurred
    //syncMutants();


    //MPI_Barrier();

// Write out population to file if needed

// Swap the population arrays
    //unsigned char *dummy=NULL;



  //}


// Record the final time
  double t2;
  if(0==rank)
  {
    t2 = MPI_Wtime();
      // Print out the final time
    printf("The total runtime for GA is %4.5f\n", t2-t1);
  }



// Finalize the MPI environment.
  MPI_Finalize();

// Free the allocated memory
  
// Create a function to do this on the GPU (or copy from previous code)
  return 0;

}

























// Function to create the initial random population

void popInit(int num_ranks, int rank, int cities, int popSize){
  
  int localPopSize = popSize/num_ranks;
  

  // Counter over the individual number
  int i = 0;
  // Counter over the gene portion to fill
  int j = 0;
  // Counter over the number to extract from log
  int k = 0;
  // random dample
  int rand_sample = 0;
  // Seed the random number generator
  time_t t = time(NULL);
  printf("time() %lld\n",(long long) t);
  srand(t*rank);
  // log to keep track so that cities are not repeated
  unsigned char *sample_log=NULL;
  sample_log = calloc(cities, sizeof(unsigned char));

  // Loop over the population that the rank is responsible for
  for( i = rank*localPopSize; i<(rank+1)*localPopSize ; i++ ){
    //printf("\n\nGenerating individual %d\n",i);

    

    // Clear out the sample log
    for(k=0; k<cities; k++){
      sample_log[k] = 0;
    }

    // Loop that generates a genome
    j = 0;
    // Fill out the gene portions
    while(j<cities){
      
      //printf("Generating city %d\n",j);
      
      // Randomly sample from the array
      rand_sample = rand()%cities;
      k = rand_sample;
      //printf("Sampled %d\n",k);
      // Check if the number has already been sampled
      while(1){
	if(!sample_log[k%cities]){
	  // Number has not been sampled
	  //printf("Number was not sampled %d\n",k);
	  // Update the log
	  sample_log[k%cities] = 1;
	  // Assign k to jth value in currentGen
	  currentGen[cities*i + j] = k%cities;
	  // Break out of the loop
	  break;
	}
	else{
	  k = (k+1)%cities;
	  //printf("Number was already sampled, moving on to %d\n",k);
	}
      }
      j = j+1;
      


    }
  }

  for( i = rank*localPopSize ; i<(rank+1)*localPopSize ; i++ ){
    printf("Individual %d: ",i);
    for(j = 0 ; j < cities ; j++){
      printf(" %d",currentGen[cities*i + j]);
    }
    printf("\n");
  }

  free(sample_log);


  

}



// Synchronize the entire population between the ranks
// This is expensive and is only needed after the initial generation
// of the population.
void syncPop(int num_ranks, int rank){
  // We can simply pass the entire array to the function.
  // Even if there is overwriting on the local rank's portion, it will
  // not matter.

  int localPopSize = popSize/num_ranks;

  unsigned char *sendbuf;
  sendbuf = &currentGen[rank*localPopSize*cities];
  int sendcount = localPopSize*cities;
  unsigned char *recvbuf;
  recvbuf = &nextGen[0];
  int recvcount = localPopSize*cities;
  

  MPI_Allgather(sendbuf, sendcount, MPI_UNSIGNED_CHAR,
	       recvbuf, recvcount, MPI_UNSIGNED_CHAR,
                 MPI_COMM_WORLD);

  unsigned char *tmp;
  tmp = currentGen;
  currentGen = nextGen;
  nextGen = tmp;


  
}



// Function the returns the number of cities
// and populates the globalmatrix containing the cost between cities

int file_read(char* file_name)
{
  FILE *fp;
  char *input = NULL, *digits,*red;
  char c;
  int k,city_idx,i,j;
  double cord[2],pos[CITY_MAX],loc;
  int num_cities = 0;
  double **mat;
  input = file_name;
  digits = (char*)calloc(20,sizeof(char));

  fp = fopen(input,"r");
  k = 0;
  //fgets reads upto the 1st blank space
    do
      {
	c = fgetc(fp);
	if(!isspace(c))
	  {
	    digits[k++] = c;
	  }
	else
	  {
	    if(k!=0)
	      { digits[k] = '\0';
		k = 0;
		printf("%s\n",digits);
		pos[num_cities++] = strtod(digits,NULL);
	      }
	  }
      }while(!feof(fp));

    //printf("%d\n",num_cities);
    //printf("%s\n",input);
    //printf("%s\n",output);
    for(i=0;i<num_cities;i++)
      printf("Here %lf\n",pos[i]);
    num_cities/=2;

    mat = distance_cities(pos,num_cities);

    mapAlloc(num_cities);

    printf("Allocate Map on GPU\n");
    

    for (i=0;i<num_cities;i++){
      for (j=0;j<num_cities;j++){
	map[i*num_cities + j] = mat[i][j];
      }
    }

    printf("copied mat into map\n");

    free(mat);

    printf("completed file read function\n");

    return num_cities;

}



// Calculates the distance between cities given the x and y
// coordinates

double** distance_cities(double* pos,int num)
{
  double *xy[num], **dist_mat;
  int i,j = 0;
  dist_mat = calloc(num,sizeof(double));

  for(i=0;i<num;i++)
    {
      xy[i] = calloc(2,sizeof(double));
      dist_mat[i] = calloc(num,sizeof(double));
    }

  for(i = 0; i<2*num;i++)
    {
      if(i%2==1)
        xy[j++][1] = pos[i];
      else
        xy[j][0] = pos[i];

    }

  for(i=0;i<num;i++)
    {
      printf("Matrix %lf %lf\n",xy[i][0],xy[i][1]);
    }

  for(i = 0;i<num;i++)
    {
      for(j = i+1;j<num;j++)
	{

          dist_mat[i][j] = sqrt(pow((xy[i][0]-xy[j][0]),2)+pow((xy[i][1]-xy[j][1]),2));
          dist_mat[j][i] = dist_mat[i][j];
	}

    }

  printf("Complete distance_cities\n");

  return dist_mat;
}






void calculateCum(int popSize, int num_ranks){
  int localPopSize = popSize/num_ranks;
  int i;
  double counter=0;
  

  for (i = 0; i < localPopSize; i++){
    counter += fitness[i];
    localCum[i] = counter;
  }
}












// Synchronize the cumulative fitness arrays
void syncCum(int num_ranks, int rank){

  int localPopSize = popSize/num_ranks;

  

  double *sendbuf;
  sendbuf = &localCum[0];
  int sendcount = localPopSize;
  double *recvbuf;
  recvbuf = &cumFitness[0];
  int recvcount = localPopSize;
  

  MPI_Allgather(sendbuf, sendcount, MPI_DOUBLE,
	       recvbuf, recvcount, MPI_DOUBLE,
                 MPI_COMM_WORLD);


  
}




void populateOffsets(int num_ranks, int popSize){
  unsigned long long localPopSize = popSize/num_ranks;
  
  int i;
  double sum = 0;
  
  
  for (i = 1; i < num_ranks+1; i++){
    sum  += cumFitness[i*localPopSize-1];
    offsets[i-1] = sum;
  }


}




void modifyCumulative(int num_ranks, int popSize){
  int localPopSize = popSize/num_ranks;
  int i = 0;
  int j = 0;

  for (i = 1; i < num_ranks; i++){
    for (j=0; j< localPopSize; j++){
      cumFitness[localPopSize*i +j] = offsets[i-1] + cumFitness[localPopSize*i +j];
    }
  }
}







void rouletteIndices(int r_wheel_length, int rank)
{  
  //r_wheel_length is the number of random numbers being generated per rank
  // Perform the copying using a kernel
  int high,low,mid,i,j,upper = cumFitness[popSize-1],lower = 0;
  double r_num;
  
  rouletteWheel = calloc(r_wheel_length,sizeof(unsigned long long));
  srand(time(0)*rank);
  for(i = 0;i<r_wheel_length;i++)
    {
      r_num = (rand()%(upper-lower+1))+lower;
      low = 0;
      high = popSize-1;
      while(low<high)
	{
	  mid = (low+high)/2;
	  if(cumFitness[mid]<r_num)
	    {
	      if(cumFitness[mid+1]>=r_num)
		{
		  if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid+1);
		  for(j = 0;j<cities;j++)
		    nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid+1)*cities + j];
		  rouletteWheel[i] = mid+1;
		  break;
		}
	      else
		{
		  if((mid+1)==popSize-1)
		    {
		      if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid+1);
		      for(j = 0;j<cities;j++)
			nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid+1)*cities + j];
		      rouletteWheel[i] = mid+1;
		      break;
		    }
		  else
		    low = mid;
		}
	    }
	  else
	    {
	      if(cumFitness[mid-1]<=r_num)
		{
		  if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid);
		  for(j = 0;j<cities;j++)
		    nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[mid*cities + j];
		  rouletteWheel[i] = mid;
		  break;
		}
	      else
		{
		  if((mid-1)==0)
		    {
		      if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid-1);
		      for(j = 0;j<cities;j++)
			nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid-1) + j];
		      rouletteWheel[i] = mid-1;
		      break;
		    }
		  else
		    high = mid;
		}
	      
	    }
	  
	}
      
    }
  
}







// Synchronize the roulette wheel indices across ranks
void syncRoulette(int num_ranks,int r_wheel_length){
  

  

  unsigned long long *sendbuf;
  sendbuf = &rouletteWheel[0];
  int sendcount = r_wheel_length;
  unsigned long long *recvbuf;
  recvbuf = &globalRouletteWheel[0];
  int recvcount = r_wheel_length;
  

  MPI_Allgather(sendbuf, sendcount, MPI_UNSIGNED_LONG_LONG,
	       recvbuf, recvcount, MPI_UNSIGNED_LONG_LONG,
                 MPI_COMM_WORLD);
  
}















