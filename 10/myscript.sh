rm ga-cuda-mpi-exe
make
./cleanup.sh
sbatch --partition=dcs -N 1 --ntasks-per-node=6 --gres=gpu:6 -t 30 ./runCode.sh
