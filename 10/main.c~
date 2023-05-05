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
// Parents and Cuts Index List
unsigned long long *samplingList;
// Mutation data
int *localMutationData;
int *globalMutationData;
int count_mutants;
int *mutation_counts_at_ranks;
// Min Fitness arrays at ranks
double *maxAtRank;
int *maxIndexAtRank;
double globalMax;
int globalMaxIndex;

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

bool parents_kernelLaunch (int num_ranks, int r_wheel_length);

void samplingAlloc(int num_ranks, int local_children_num);

void populateSamplingList(int local_children_num, int num_ranks, int r_wheel_length);

bool crossover_kernelLaunch (int num_ranks, int rank, int local_children_num);

void syncChildren(int num_ranks, int rank, int popSize, int local_children_num);

void createLocalMutation(int num_ranks, int rank, double mutationRatio);

void syncMutationSizes(int num_ranks,int rank);

void syncMutationData(int num_ranks, int rank);

void performMutation(int num_ranks);

void findMax(int popSize,int num_ranks,int rank);

void syncMaxs(int num_ranks, int rank);

void findGlobalMax(int num_ranks, int rank);













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

  // Dummy variable to hold data while swapping arrays
  unsigned char *dummy=NULL;

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

  samplingAlloc(num_ranks, local_children_num);

  mutation_counts_at_ranks = calloc(num_ranks, sizeof(int));



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

  /*
  if(0==rank){
    for( i = 0 ; i<popSize ; i++ ){
      printf("Individual %d: ",i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",currentGen[cities*i + j]);
      }
      printf("\n");
    }
  }
  */
  
  printf("Allocating fitness array on GPU\n");
  // Allocate fitness array
  fitnessAlloc(num_ranks, rank, popSize);
  localCum = calloc(popSize/num_ranks, sizeof(double));
  cumFitness = calloc(popSize, sizeof(double));
  offsets = calloc(num_ranks, sizeof(double));
  globalRouletteWheel = calloc(r_wheel_length*num_ranks,sizeof(unsigned long long));

  maxAtRank = calloc(num_ranks,sizeof(double));
  maxIndexAtRank = calloc(num_ranks,sizeof(int));


  // Run loop till converged
  int iter=0;
  while (iter<1000){
    iter++;

    // Calculate Fitness Values
    //printf("Allocated fitness array on GPU\n");
    //printf("Launching fitness kernel\n");
    fitness_kernelLaunch(num_ranks, rank, popSize, cities);
    //printf("Launched fitnedd kernel\n");
    
    
    /*
    for(i=0;i<cities;i++){
      for(j=0;j<cities;j++){
	printf("%f ",map[i*cities +j]);
      }
      printf("\n");
    }
    */

    
    //printf("Fitness of Individual 0 %lf \n",fitness[0]);
    /*
    //printf("Printing fitness array\n");
    
    for(i = 0; i < popSize/num_ranks; i++){
    //printf("Fitness of Individual %llu : %f\n",rank*popSize/num_ranks+i,fitness[i]);
    }
    */

    
    //printf("Calculating the cumulative fitness \n");
    // Calculate the cumulative fitness on the CPU; will be faster on the CPU
    calculateCum(popSize, num_ranks);
    
    //printf("Printing the local Cumulative sum\n");
    
    
    /*
      for(i = 0; i < popSize/num_ranks; i++){
      //printf("Cumulative sum at Individual %llu : %f\n",rank*popSize/num_ranks+i,localCum[i]);
      }
    */
    
    
    
    //printf("Synchronizing the cumulative sums\n");
    // Synchronize the cumulative fitness arrays localCum to cumFitness
    syncCum(num_ranks,rank);
    /*
      if (0==rank){
      printf("Printing the cumulative sums\n");
      for(i = 0; i < popSize; i++){
      printf("Individual %d : %f\n", i,cumFitness[i]);
      }
      }
    */
    
    
    
    //printf("Populating offsets\n");
    populateOffsets(num_ranks, popSize);
    /*
    if (0==rank){
    printf("Printing the offsets\n");
    for(i = 0; i < num_ranks; i++){
    printf("Rank %d : %f\n", i,offsets[i]);
    }
    }
  */
  

  modifyCumulative(num_ranks, popSize);
  /*
    if (0==rank){
    printf("Printing modified cumulative sums\n");
    for(i = 0; i < popSize; i++){
    printf("Individual %d : %f\n", i,cumFitness[i]);
    }
    }  
  */
  
  /*
    if(0==rank){
    printf("Generating the Roulette Wheel\n");
    }
  */
  // Generate the roulette wheel for this rank
  rouletteIndices(r_wheel_length, rank);

  /*
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
  */

  syncRoulette(num_ranks,r_wheel_length);

  /*
  if(0==rank){
    printf("Printing the global roulette wheel\n");
    for(i=0;i<num_ranks*r_wheel_length; i++){
      printf("%llu \n",globalRouletteWheel[i]);
    }
  }
  */
  
  parents_kernelLaunch(num_ranks, r_wheel_length);
  
  /*
    if(0==rank){
    printf("Next gen individuals after roulette selection kernel launch\n");
    for( i = 0 ; i<popSize ; i++ ){
    printf("Next Individual after kernel %d: ",i);
    for(j = 0 ; j < cities ; j++){
    printf(" %d",nextGen[cities*i + j]);
    }
    printf("\n");
    }
    } 
  */
  
  // Create the list of parents that will generate the children, and the cuts that will 
  // be performed
  populateSamplingList(local_children_num, num_ranks, r_wheel_length);

  /*
  if (0==rank){
    printf("Local Children number is %d \n", local_children_num);
    printf("Printing pop list\n");
    for(i=0;i<local_children_num;i++){
      printf("Child %d recipe ", i);
      for(j=0; j<4;j++){
	printf(" %llu ", samplingList[i*4+j]);
      }
      printf("\n");
    }
  }
  */

  // Launch the crossover kernel
  crossover_kernelLaunch (num_ranks, rank, local_children_num);
  

  /*
  if(0==rank){
   printf("Next gen individuals after crossover kernel launch\n");
    for( i = 0 ; i<popSize ; i++ ){
      printf("Rank %d after kernel %d: ", rank, i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",nextGen[cities*i + j]);
      }
      printf("\n");
    }
  } 
  */


  // Synchronize the children only
  syncChildren(num_ranks, rank, popSize, local_children_num);
  


  /*
  if(0==rank){
    printf("Next gen individuals after syncing children\n");
    for( i = 0 ; i<popSize ; i++ ){
      printf("Rank %d after sync %d: ", rank, i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",nextGen[cities*i + j]);
      }
      printf("\n");
    }
  } 
  */


  // Perform mutations on the CPU
  // Use MPI_allgatherv to sync the uneven numbers
  //printf("Running create local mutation\n");
  createLocalMutation(num_ranks, rank, mutationRatio);
  

  /*
  for(i=0; i<count_mutants; i++){
    printf("   Rereading rank %d,  ind %d     ind1 %d    ind2 %d \n", rank,
	   localMutationData[i*3],
	   localMutationData[i*3+1],
	   localMutationData[i*3+2]
	   );
  }
  */
  
  // MPI_allgather sizes of mutations
  // Synchronize all the mutation data using mpi allgatherv
  //printf("Running sync local mutation\n");
  //syncMutationData(num_ranks, rank);
  syncMutationSizes(num_ranks, rank);

  /*
  if(0==rank){
    for(i=0; i< num_ranks; i++){
      printf("Mutation count of rank %d is %d \n", i, mutation_counts_at_ranks[i]);
    }
  }
  */

  
  //printf("Calling syncmuationdata\n");

  syncMutationData(num_ranks, rank);

  //printf("Performing Mutations\n");
  performMutation(num_ranks);


  /*
  if(0==rank){
    printf("Next gen individuals after mutation\n");
    for( i = 0 ; i<popSize ; i++ ){
      printf("Rank %d after mut %d: ", rank, i);
      for(j = 0 ; j < cities ; j++){
	printf(" %d",nextGen[cities*i + j]);
      }
      printf("\n");
    }
  } 
  */



  // Finding the minimum of the ranks
  findMax(popSize, num_ranks, rank);
  // Synchronizing the minimums at each rank and their indices
  syncMaxs(num_ranks,rank);


  /*  
  if(0==rank){
    for(i=0;i<num_ranks;i++){
      printf("             Reading out synced rank %d     min is %f      at index     %d\n",i, minAtRank[i], minIndexAtRank[i]);
    }
  }
  */

  
  // Find the Global Index based on the syned minimums and indices
  findGlobalMax(num_ranks, rank);


  if(0==rank){
    printf("The global minimum at iter %d is %lf  at index %d \n",iter, 1.0/globalMax, globalMaxIndex);
  }

  // Write out population to file if needed
  
  // Swap the population arrays
  dummy = currentGen;
  currentGen = nextGen;
  nextGen = dummy;
    



  }


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

  free(mutation_counts_at_ranks);
  free(localCum);
  free(cumFitness);
  free(offsets);
  free(globalRouletteWheel);
  free(rouletteWheel);
  free(localMutationData);
  free(globalMutationData);


  free(maxAtRank);
  free(maxIndexAtRank);


  
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
		//printf("%s\n",digits);
		pos[num_cities++] = strtod(digits,NULL);
	      }
	  }
      }while(!feof(fp));

    //printf("%d\n",num_cities);
    //printf("%s\n",input);
    //printf("%s\n",output);
    //for(i=0;i<num_cities;i++)
    //printf("Here %lf\n",pos[i]);
    num_cities/=2;

    mat = distance_cities(pos,num_cities);

    mapAlloc(num_cities);

    //printf("Allocate Map on GPU\n");
    

    for (i=0;i<num_cities;i++){
      for (j=0;j<num_cities;j++){
	map[i*num_cities + j] = mat[i][j];
      }
    }

    //printf("copied mat into map\n");

    free(mat);
    free(digits);

    //printf("completed file read function\n");

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
      //printf("Matrix %lf %lf\n",xy[i][0],xy[i][1]);
    }

  for(i = 0;i<num;i++)
    {
      for(j = i+1;j<num;j++)
	{

          dist_mat[i][j] = sqrt(pow((xy[i][0]-xy[j][0]),2)+pow((xy[i][1]-xy[j][1]),2));
          dist_mat[j][i] = dist_mat[i][j];
	}

    }

  //printf("Complete distance_cities\n");
  
  //free(xy);

  return dist_mat;
}






void calculateCum(int popSize, int num_ranks){
  int localPopSize = popSize/num_ranks;
  int i;
  double counter=0.0;
  

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
  double sum = 0.0;
  
  
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
		  //if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid+1);
		  //for(j = 0;j<cities;j++)
		    //nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid+1)*cities + j];
		  rouletteWheel[i] = mid+1;
		  break;
		}
	      else
		{
		  if((mid+1)==popSize-1)
		    {
		      //if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid+1);
		      //for(j = 0;j<cities;j++)
			//nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid+1)*cities + j];
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
		  //if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid);
		  //for(j = 0;j<cities;j++)
		   // nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[mid*cities + j];
		  rouletteWheel[i] = mid;
		  break;
		}
	      else
		{
		  if((mid-1)==0)
		    {
		      //if(0==rank) printf("Number is  %lf with index %d\n",r_num,mid-1);
		      //for(j = 0;j<cities;j++)
			//nextGen[(rank*r_wheel_length+i)*cities + j] = currentGen[(mid-1) + j];
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









void populateSamplingList(int local_children_num, int num_ranks, int r_wheel_length){
  int i;

  unsigned long long r1,r2;

  for(i=0;i<local_children_num*4;i+=4){

    //printf("i = %d\n",i);

    // First fill in the patents from gloabl roulette wheel
    r1 = rand()%(r_wheel_length*num_ranks);
    r2 = rand()%(r_wheel_length*num_ranks); 
    //printf("The two selected roulette wheel indices are %llu and %llu \n",r1, r2 );
    
    samplingList[i] = globalRouletteWheel[r1];
    samplingList[i+1] = globalRouletteWheel[r2];

        
    // Now fill in the cuts using randm cities; do not make cuts equal
    // Cut 1 < Cut 2
    unsigned long long cut1;
    unsigned long long cut2;
    while(1){
      cut1 = rand()%cities;
      cut2 = rand()%cities;
      if (cut1 < cut2){
	samplingList[i+2] = cut1;
	samplingList[i+3] = cut2;
	break;
      }
      
      
    }
    //printf("THe two cuts are %llu, %llu \n", cut1, cut2);
    
  }
}









void syncChildren(int num_ranks, int rank, int popSize, int local_children_num){

  int sendcount = local_children_num * cities;
  int *recvcounts;
  int *displs;
  int i;
  int localOffset = ( (rank+1)*(popSize/num_ranks) - local_children_num )*cities;
		     
  //calloc recvcounts
  recvcounts = calloc(num_ranks, sizeof(int));
  displs = calloc(num_ranks, sizeof(int));
  
  for(i=0; i<num_ranks; i++){
    recvcounts[i] = local_children_num*cities;
    displs[i] = ( (i+1)*(popSize/num_ranks) - local_children_num )*cities;
  }

  MPI_Allgatherv( &nextGen[localOffset], sendcount, MPI_UNSIGNED_CHAR,
		  &nextGen[0], recvcounts, displs,
		  MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

  //  MPI_Barrier(MPI_COMM_WORLD);

  free(recvcounts);
  free(displs);

}









void createLocalMutation(int num_ranks, int rank, double mutationRatio){
  
  int i;
  count_mutants=0;

  int index1, index2, individual;

  

  for(i = 0; i <popSize/num_ranks; i++){
    if(rand()%(100) < mutationRatio*100){
      count_mutants+=1;
    }  
  }


  localMutationData = calloc(count_mutants*3, sizeof(int));
  //printf("     Rank %d created %d mutants\n", rank, count_mutants);

  

  for(i = 0; i< count_mutants; i++)
    {
      index1 = rand()%cities;
      while(1){
	index2 = rand()%cities;
	if (index1!=index2){
	  break;
	}
      }
      individual = rand()%(popSize);
      
      localMutationData[i*3] = individual;
      localMutationData[i*3+1] = index1;
      localMutationData[i*3+2] = index2;    
      /*printf("Rank %d,    Ind  %d     ind1 %d     ind2 %d\n", rank,
	     localMutationData[i*3],
	     localMutationData[i*3+1],
	     localMutationData[i*3+2]      
	     );
      */
    }
  

}










void syncMutationSizes(int num_ranks,int rank){

  MPI_Allgather(&count_mutants, 1 , MPI_INT,
		mutation_counts_at_ranks, 1, MPI_INT,
		MPI_COMM_WORLD);
}





void syncMutationData(int num_ranks, int rank){
  
  // send buffer is locamutationdata

  int gmsize=0;
  int i,j;
  for(i=0; i<num_ranks; i++){
    gmsize += mutation_counts_at_ranks[i]*3;
  }
  globalMutationData = calloc(gmsize, sizeof(int));

  
  // Create the displacements array
  int *displacements;
  int *dataLengths;
  displacements = calloc(num_ranks, sizeof(int));
  dataLengths = calloc(num_ranks, sizeof(int));  
  
  int counter = 0;
  for(i=0; i<num_ranks; i++){
    displacements[i] = counter;    
    counter += mutation_counts_at_ranks[i]*3;
  }

  for(i=0; i<num_ranks; i++){
    dataLengths[i] =  mutation_counts_at_ranks[i]*3;
  }
  

  MPI_Allgatherv(localMutationData, mutation_counts_at_ranks[rank]*3 , MPI_INT,
		     globalMutationData, dataLengths, displacements,
		     MPI_INT, MPI_COMM_WORLD);

  /*
  if(0==rank){
    for(i=0;i<num_ranks;i++){
      for(j=0;j<mutation_counts_at_ranks[i];j++){
	printf("Synced rank %d   indiv %d    inde %d  inde %d\n",i,
	       globalMutationData[displacements[i] + 3*j],
	       globalMutationData[displacements[i] + 3*j +1],
	       globalMutationData[displacements[i] + 3*j +2]
	       );
      }
    }
  }
  */

  free(displacements);
  free(dataLengths);
}



void performMutation(int num_ranks){
  
  // count the total number of mutations
  int total = 0;
  
  int i;
  for(i=0;i<num_ranks;  i++){
    total += mutation_counts_at_ranks[i];
  }


  int individual;
  int index1,index2;
  unsigned char temp;
  //Perform the mutations
  for (i=0; i<total; i++){
    individual = globalMutationData[i*3];
    index1 = globalMutationData[i*3+1];
    index2 = globalMutationData[i*3+2];

    temp = nextGen[individual*cities +index1];
    nextGen[individual*cities +index1] = nextGen[individual*cities +index2];
    nextGen[individual*cities +index2] = temp;
  }


}





void findMax(int popSize, int num_ranks, int rank){
  
  int i;
  double currentmax=fitness[0];
  int maxindex = 0;
  for (i=0; i< popSize/num_ranks; i++){
    if (currentmax < fitness[i]){
      currentmax = fitness[i];
      maxindex = i;
    }
  }
  maxAtRank[rank] = currentmax;
  maxIndexAtRank[rank] = maxindex + rank*popSize/num_ranks;

  //printf("At rank %d     min is %f      at index     %d\n",rank,currentmin,minindex);
}



void syncMaxs(int num_ranks, int rank){

  MPI_Allgather(&maxAtRank[rank], 1, MPI_DOUBLE,
		maxAtRank, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  
  MPI_Allgather(&maxIndexAtRank[rank], 1, MPI_INT,
		maxIndexAtRank, 1, MPI_INT, MPI_COMM_WORLD);

  
}



void findGlobalMax(int num_ranks, int rank){
  
  int i;
  double max = maxAtRank[0];
  int maxindex = maxIndexAtRank[0];
  
  for(i=0; i<num_ranks; i++){
    if (max < maxAtRank[i]){
      max = maxAtRank[i];
      maxindex = maxIndexAtRank[i];
    }
  }
  
  globalMax = max;
  globalMaxIndex = maxindex;
  
}
