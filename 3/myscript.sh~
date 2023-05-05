rm ga-cuda-mpi-exe
make
./cleanup.sh
sbatch --partition=dcs -N 1 --ntasks-per-node=1 --gres=gpu:1 -t 30 ./runCode.sh
