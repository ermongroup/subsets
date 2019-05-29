#!/bin/sh
cd ../subsets/knn
export PATH="$PATH:/usr/local/cuda-9.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64"
pwd; hostname; date

for tau in 0.1 1 5 16 64
do
    python run_dknn.py --k 9 --tau ${tau} --method subsets --dataset mnist
    python run_dknn.py --k 9 --tau ${tau} --method subsets --dataset fashion-mnist
    python run_dknn.py --k 9 --tau ${tau} --method subsets --dataset cifar10
done

date

