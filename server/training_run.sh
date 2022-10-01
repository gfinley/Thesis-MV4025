#!/bin/bash
#SBATCH -N 1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=50
#SBATCH --output=Navy_DQN_test_%j.txt
#SBATCH --time=15-23:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthew.finley@nps.edu






#python server.py city-inf-5 --blueAI Lab3_dqn-v3_500000_0 --redAI pass-agg --nReps 25000 > Data/data_Lab3_dqn-v3_500K_0.txt
#python server.py city-inf-5 --blueAI Lab3_dqn-v3_2000000_0 --redAI pass-agg --nReps 25000 > Data/data_Lab3_dqn-v3_2M_0.txt

python ray_run.py --name $5 --worker_num $1 --worker_cpu $2 --driver_cpu $3 --algo $4

#python train_lab5.py --model mod_1 --length 2000000