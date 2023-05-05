rm ga-cuda-mpi-exe
make
./cleanup.sh
sbatch -N 1 --ntasks-per-node=1 --gres=gpu:1 -t 30 ./runCode.sh
sbatch -N 1 --ntasks-per-node=2 --gres=gpu:2 -t 30 ./runCode.sh
sbatch --partition=dcs -N 1 --ntasks-per-node=6 --gres=gpu:6 -t 30 ./runCode.sh
sbatch --partition=dcs -N 2 --ntasks-per-node=6 --gres=gpu:6 -t 30 ./runCode.sh
