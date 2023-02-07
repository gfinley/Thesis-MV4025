#!/bin/bash
#SBATCH -N 1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=0_bayesian_test_%j.txt
#SBATCH --time=15-23:00:00




#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthew.finley@nps.edu

#for running training
python ray_run_3.py --name $5 --worker_num $1 --worker_cpu $2 --driver_cpu $3 --algo $4 --model $6 --gpu $7



#FOR EVALUATION
#python ray_eval.py --checkpoint /home/matthew.finley/Thesis-MV4025/3_IMPALA_parallel_gpu/IMPALA_30_1_8__gpu_5M/IMPALA_atlatl_a4f67_00000_0_2023-01-08_15-17-39/checkpoint_000177 --driver_cpu 10

#for infrence
#python ray_infrence_test.py