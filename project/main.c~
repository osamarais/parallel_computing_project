// Genetic Algorithm on Multiple GPUs

// Project submission for CSCI 6360 RPI
// Osama M. Raisuddin
// Avinash Moharana
// Shaunak Basu




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

#define CITY_MAX 10000


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
// Allocate population arrays
void popAlloc(int cities, int popSize);
// Allocate map matrix
void mapAlloc(unsigned long long cities);
// Allocate fitness array
void fitnessAlloc(int num_ranks, int rank, int popSize);


// Protype of functions
double** distance_cities(double*, int);
// Read in map file
int file_read(char* file_name);

// Initialize a random population
void popInit(int num_ranks, int rank, int cities, int popSize);
// Synchronize the initial random population among all ranks
void syncPop(int num_ranks, int rank);
// Calculate the fitness of each individual
bool fitness_kernelLaunch (int num_ranks, int rank, int popSize, int cities);
// Calculate the cumulative fitness at each rank
void calculateCum(int popSize, int num_ranks);
// synchronize the cumulative fitnesses at each rank
void syncCum(int num_ranks, int rank);
// create the array that contains the cumulative fitness of each array
void populateOffsets(int num_ranks, int popSize);
void modifyCumulative(int num_ranks, int popSize);
// Use roulette selection to select next generation
void rouletteIndices(int r_wheel_length, int rank);
// Share the indices of the selected individuals across ranks
void syncRoulette(int num_ranks,int r_wheel_length);
// Kernel to copy the selected parents
bool parents_kernelLaunch (int num_ranks, int r_wheel_length);
// Allocate the arrays that hold indices of sampled parents
void samplingAlloc(int num_ranks, int local_children_num);
// Sample parents
void populateSamplingList(int local_children_num, int num_ranks, int r_wheel_length);
// Crossover parents to create children
bool crossover_kernelLaunch (int num_ranks, int rank, int local_children_num);
// Synchronize children among ranks
void syncChildren(int num_ranks, int rank, int popSize, int local_children_num);
// Create mutation recipes on each rank
void createLocalMutation(int num_ranks, int rank, double mutationRatio);
// Synchronize recipes among ranks
void syncMutationSizes(int num_ranks,int rank);
void syncMutationData(int num_ranks, int rank);
// Perform mutation using the mutation recipes
void performMutation(int num_ranks);
// Find max fitness within rank
void findMax(int popSize,int num_ranks,int rank);
// Synchronize the maximum fitnesses for each rank
void syncMaxs(int num_ranks, int rank);
// Find the global max fitness value
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
  
  // Set the algorithm parameters
  double crossoverRatio=1;
  double mutationRatio=1;

  // Dummy variable to hold data while swapping arrays
  unsigned char *dummy=NULL;
  
  // Read in the file, get the number of cities
  cities =  file_read(filename);
  
  // Initialize cities^2 population size
  popSize = ((cities*cities)/num_ranks)*num_ranks;
  // Now we want to set the GPU device for each rank, and initialize the worlds
  popAlloc(cities, popSize);
  // generate the rank's portion of the initial population
  popInit(num_ranks, rank, cities, popSize);
  
  // Calculate the number of parents to be sampled based on crossover ratios
  int r_wheel_length = floor(crossoverRatio*popSize/num_ranks);
  // Calculate the number of children to create from parents
  int local_children_num = popSize/num_ranks - r_wheel_length;  
  // Allocate memory for the sampling arrays
  samplingAlloc(num_ranks, local_children_num);
  // Allocate memory for mutation recipes
  mutation_counts_at_ranks = calloc(num_ranks, sizeof(int));
  

  // Synchronize the initial population across ranks
  syncPop(num_ranks, rank);

  
  MPI_Barrier(MPI_COMM_WORLD);

  int i = 0;
  int j = 0;

  
  // Allocate dynamically sized arrays
  fitnessAlloc(num_ranks, rank, popSize);
  localCum = calloc(popSize/num_ranks, sizeof(double));
  cumFitness = calloc(popSize, sizeof(double));
  offsets = calloc(num_ranks, sizeof(double));
  globalRouletteWheel = calloc(r_wheel_length*num_ranks,sizeof(unsigned long long));
  maxAtRank = calloc(num_ranks,sizeof(double));
  maxIndexAtRank = calloc(num_ranks,sizeof(int));
  
  
  // Run loop till converged
  int iter=0;
  while (iter<10000){
    iter++;
    
    // Calculate Fitness Values
    fitness_kernelLaunch(num_ranks, rank, popSize, cities);
    // Calculate the cumulative fitness on the CPU; will be faster on the CPU
    calculateCum(popSize, num_ranks);
    // Synchronize the cumulative fitness arrays localCum to cumFitness
    syncCum(num_ranks,rank);
    populateOffsets(num_ranks, popSize);
    modifyCumulative(num_ranks, popSize);
    // Generate the roulette wheel for this rank
    rouletteIndices(r_wheel_length, rank);
    syncRoulette(num_ranks,r_wheel_length);
    parents_kernelLaunch(num_ranks, r_wheel_length);
    // Create the list of parents that will generate the children, and the cuts that will 
    // be performed
    populateSamplingList(local_children_num, num_ranks, r_wheel_length);
    // Launch the crossover kernel
    crossover_kernelLaunch (num_ranks, rank, local_children_num);
    // Synchronize the children only
    syncChildren(num_ranks, rank, popSize, local_children_num);
    // Perform mutations on the CPU
    // Use MPI_allgatherv to sync the uneven numbers
    createLocalMutation(num_ranks, rank, mutationRatio);
    // MPI_allgather sizes of mutations
    // Synchronize all the mutation data using mpi allgatherv
    syncMutationSizes(num_ranks, rank);
    syncMutationData(num_ranks, rank);
    // Perform mutations using the synchronized recipes
    performMutation(num_ranks);
    // Finding the minimum of the ranks
    findMax(popSize, num_ranks, rank);
    // Synchronizing the minimums at each rank and their indices
    syncMaxs(num_ranks,rank);
    // Find the Global Index based on the syned minimums and indices
    findGlobalMax(num_ranks, rank);
    
    if(0==rank){
      printf("The global minimum at iter %d is %lf  at index %d \n",iter, 1.0/globalMax, globalMaxIndex);
    }
    
    // Write out population to file if flag is set
  
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
  
  // Free the dynamicaly allocated memory
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
    
    // Clear out the sample log
    for(k=0; k<cities; k++){
      sample_log[k] = 0;
    }
    
    // Loop that generates a genome
    j = 0;
    // Fill out the gene portions
    while(j<cities){
      
      // Randomly sample from the array
      rand_sample = rand()%cities;
      k = rand_sample;
      // Check if the number has already been sampled
      while(1){
	if(!sample_log[k%cities]){
	  // Number has not been sampled
	  // Update the log
	  sample_log[k%cities] = 1;
	  // Assign k to jth value in currentGen
	  currentGen[cities*i + j] = k%cities;
	  // Break out of the loop
	  break;
	}
	else{
	  // Number was already sampled
	  k = (k+1)%cities;
	}
      }
      j = j+1;
      
      

    }
  }
  
  free(sample_log);
  
  
  
  
}



// Synchronize the entire population between the ranks
// This is expensive and is only needed after the initial generation
// of the population.
void syncPop(int num_ranks, int rank){
 
  int localPopSize = popSize/num_ranks;

  unsigned char *sendbuf;
  sendbuf = &currentGen[rank*localPopSize*cities];
  int sendcount = localPopSize*cities;
  unsigned char *recvbuf;
  recvbuf = &nextGen[0];
  int recvcount = localPopSize*cities;
  
  // Allgather the information to store data in nextgen
  MPI_Allgather(sendbuf, sendcount, MPI_UNSIGNED_CHAR,
	       recvbuf, recvcount, MPI_UNSIGNED_CHAR,
                 MPI_COMM_WORLD);

  // swap pointers of current and next gen
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
		pos[num_cities++] = strtod(digits,NULL);
	      }
	  }
      }while(!feof(fp));

    num_cities/=2;

    mat = distance_cities(pos,num_cities);

    mapAlloc(num_cities);

    for (i=0;i<num_cities;i++){
      for (j=0;j<num_cities;j++){
	map[i*num_cities + j] = mat[i][j];
      }
    }

    free(mat);
    free(digits);

    return num_cities;

}


// Calculates the distance between cities given the x and y
// coordinates
// Only runs once to load the matrix of city distances

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

  for(i = 0;i<num;i++)
    {
      for(j = i+1;j<num;j++)
	{

          dist_mat[i][j] = sqrt(pow((xy[i][0]-xy[j][0]),2)+pow((xy[i][1]-xy[j][1]),2));
          dist_mat[j][i] = dist_mat[i][j];
	}
    }

  return dist_mat;
}





// Calculates the cumulative sum of the fitness values for
// roulette selection
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
// across ranks using allgather
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




// Get the offsets of each portion of the cumulative fitness
void populateOffsets(int num_ranks, int popSize){
  unsigned long long localPopSize = popSize/num_ranks;
  
  int i;
  double sum = 0.0;
   
  for (i = 1; i < num_ranks+1; i++){
    sum  += cumFitness[i*localPopSize-1];
    offsets[i-1] = sum;
  }
}



// modify the cumulative fitness values based on the offsets
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



// Get the index number for a roulette wheel selection
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
		  rouletteWheel[i] = mid+1;
		  break;
		}
	      else
		{
		  if((mid+1)==popSize-1)
		    {
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
		  rouletteWheel[i] = mid;
		  break;
		}
	      else
		{
		  if((mid-1)==0)
		    {
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







// Synchronize the roulette wheel indices that were selected across ranks
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








// Select random parents from the roulette to create children
void populateSamplingList(int local_children_num, int num_ranks, int r_wheel_length){
  int i;

  unsigned long long r1,r2;

  for(i=0;i<local_children_num*4;i+=4){

    // First fill in the patents from gloabl roulette wheel
    r1 = rand()%(r_wheel_length*num_ranks);
    r2 = rand()%(r_wheel_length*num_ranks); 
    
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
    
  }
}








// Synchronize the children created on each rank among all ranks
void syncChildren(int num_ranks, int rank, int popSize, int local_children_num){

  int sendcount = local_children_num * cities;
  int *recvcounts;
  int *displs;
  int i;
  int localOffset = ( (rank+1)*(popSize/num_ranks) - local_children_num )*cities;
		     
  recvcounts = calloc(num_ranks, sizeof(int));
  displs = calloc(num_ranks, sizeof(int));
  
  for(i=0; i<num_ranks; i++){
    recvcounts[i] = local_children_num*cities;
    displs[i] = ( (i+1)*(popSize/num_ranks) - local_children_num )*cities;
  }

  MPI_Allgatherv( &nextGen[localOffset], sendcount, MPI_UNSIGNED_CHAR,
		  &nextGen[0], recvcounts, displs,
		  MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

  free(recvcounts);
  free(displs);

}








// Generate recipes for mutations
void createLocalMutation(int num_ranks, int rank, double mutationRatio){
  
  int i;
  count_mutants=0;

  int index1, index2, individual;

  
  // get a number of mutations to perform based on random sampling and the mutation ratio
  for(i = 0; i <popSize/num_ranks; i++){
    if(rand()%(100) < mutationRatio*100){
      count_mutants+=1;
    }  
  }

  
  localMutationData = calloc(count_mutants*3, sizeof(int));
  // Generate recipes and store them
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
    }
  

}









// Synchronize the number of mutations on each rank
void syncMutationSizes(int num_ranks,int rank){

  MPI_Allgather(&count_mutants, 1 , MPI_INT,
		mutation_counts_at_ranks, 1, MPI_INT,
		MPI_COMM_WORLD);
}




// Synchronize the recipes across the ranks
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


  free(displacements);
  free(dataLengths);
}


// After recipes are shared, perform the mutations
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




// Find the maximum fitness on the current rank
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

}


// Synchronise the maximums across the ranks
void syncMaxs(int num_ranks, int rank){

  MPI_Allgather(&maxAtRank[rank], 1, MPI_DOUBLE,
		maxAtRank, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  
  MPI_Allgather(&maxIndexAtRank[rank], 1, MPI_INT,
		maxIndexAtRank, 1, MPI_INT, MPI_COMM_WORLD);

  
}


// Using synchronized maximums from each rank, get the global maximum
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
