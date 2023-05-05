rm ga-cuda-mpi-exe
make
./cleanup.sh
sbatch --partition=dcs -N 1 --ntasks-per-node=3 --gres=gpu:3 -t 30 ./runCode.sh
