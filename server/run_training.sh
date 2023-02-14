#sbatch training_run.sh 1 2 8

#for RAY
#sbatch training_run.sh 21 2 8

#Experiment 1
#sbatch training_run.sh 21 2 8 PPO FC_NET FC_NET
#sleep 1
#sbatch training_run.sh 21 2 8 PPO Model_Vision Model_Vision
#sleep 1
#sbatch training_run.sh 21 2 8 PPO Hex_CNN Hex_CNN


#Experiment 2
#sbatch training_run.sh 20 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 15 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 10 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 5 2 8 PPO FC_NET FC_NET


#Experiment 3
#changed the way ray init used total cpus,
#I suspect this is the reason performance is always the same
#sbatch training_run.sh 20 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 15 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 10 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 5 2 8 PPO FC_NET FC_NET

#EXPERIMENT 4
#THROW MORE cpu AT THE PROPBLEM
#sbatch training_run.sh 45 2 10 PPO Hex_CNN Hex_CNN

#Experiment 5
#set number of envs per worker to 2. 2 CPUs per worker
#sbatch training_run.sh 45 2 10 PPO Hex_CNN Hex_CNN
#sbatch training_run.sh 45 2 10 PPO Hex_CNN Hex_CNN

#Experiment 6
#Run ray with only 4 CPUs
#sbatch training_run.sh 0 0 4 PPO BASELINE Hex_CNN

#Experiment 7
#Run ray with a GPU for comparison
#17-34
#sbatch training_run.sh 10 2 8 PPO Hex_CNN_GPU Hex_CNN


#Experiment 8
#Run ray with a GPU for comparison with FC net for GPU improvement
#sbatch training_run.sh 10 2 8 PPO FC_NET FC_NET
#sbatch training_run.sh 10 2 8 PPO CNN Model_Vision
#sbatch training_run.sh 32 1 8 PPO HEX_GPU Hex_CNN

##expirement 9

#
#
#expirement 10
#sbatch training_run.sh 1 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 5 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 10 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 15 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 20 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 25 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 30 1 8 PPO PPO_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 1 1 8 PPO IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 5 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 10 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 15 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 20 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 25 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN
#sbatch training_run.sh 30 1 8 IMPALA IMPALA_parallel_00_GPU Hex_CNN

#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=100 training_run.sh 45 2 10 PPO 12_Baysian_training Model_bayesian 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=100 training_run.sh 45 2 10 PPO 12_Baysian_training Hex_CNN 0
##experiment 3
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run.sh 1 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=13 training_run.sh 5 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=18 training_run.sh 10 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=23 training_run.sh 15 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=28 training_run.sh 20 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=33 training_run.sh 25 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=38 training_run.sh 30 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=43 training_run.sh 35 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=48 training_run.sh 40 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=53 training_run.sh 45 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=58 training_run.sh 50 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=63 training_run.sh 55 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=68 training_run.sh 60 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=73 training_run.sh 65 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=78 training_run.sh 70 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=83 training_run.sh 75 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=88 training_run.sh 80 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=93 training_run.sh 85 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=98 training_run.sh 90 1 8 PPO 3_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run.sh 1 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=13 training_run.sh 5 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=18 training_run.sh 10 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=23 training_run.sh 15 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=28 training_run.sh 20 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=33 training_run.sh 25 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=38 training_run.sh 30 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=43 training_run.sh 35 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=48 training_run.sh 40 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=53 training_run.sh 45 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=58 training_run.sh 50 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=63 training_run.sh 55 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=68 training_run.sh 60 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=73 training_run.sh 65 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=78 training_run.sh 70 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=83 training_run.sh 75 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=88 training_run.sh 80 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=93 training_run.sh 85 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=98 training_run.sh 90 1 8 IMPALA 3_IMPALA_parallel Hex_CNN 0
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 1 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 5 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 10 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 15 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 20 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 25 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 30 1 8 PPO 3_PPO_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 1 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 5 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 10 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 15 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 20 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 25 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 30 1 8 IMPALA 3_IMPALA_parallel_gpu Hex_CNN 1
#

##size complexity run
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-5 Hex_CNN 
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-6 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-7 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-8 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-9 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-10 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-11 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-12 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-13 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-14 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-15 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-16 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-17 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-18 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-19 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-20 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-21 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-22 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-23 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-24 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-25 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-26 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-27 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-28 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-29 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-30 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-31 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-32 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-33 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-34 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-35 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-36 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-37 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-38 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-39 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-40 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-41 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-42 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-43 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-44 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-45 Hex_CNN 
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-46 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-47 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-48 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-49 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-50 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-51 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-52 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-53 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-54 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-55 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-56 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-57 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-58 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-59 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-60 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-61 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-62 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-63 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-64 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-65 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-66 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-67 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-68 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-69 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-70 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-71 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-72 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-73 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-74 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-75 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-76 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-77 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-78 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-79 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-80 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-81 Hex_CNN 
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-82 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-83 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-84 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-85 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-86 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-87 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-88 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-89 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-90 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-91 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-92 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-93 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-94 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-95 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-96 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-97 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-98 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-99 Hex_CNN
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run.sh 42 1 8 IMPALA clear-inf-x-100 Hex_CNN



#GPU complexity Run Experiment 4
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-5 Hex_CNN 
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-6 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-7 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-8 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-9 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-10 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-11 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-12 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-13 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-14 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-15 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-16 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-17 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-18 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-19 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-20 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-21 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-22 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-23 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-24 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-25 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-26 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-27 Hex_CNN 
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-28 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-29 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-30 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-31 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-32 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-33 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-34 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-35 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-36 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-37 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-38 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-39 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-40 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-41 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-42 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-43 Hex_CNN 
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-44 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-45 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-46 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-47 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-48 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-49 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-50 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-51 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-52 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-53 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-54 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-55 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-56 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-57 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-58 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-59 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-60 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-61 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-62 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-63 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-64 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-65 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-66 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-67 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-68 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-69 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-70 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-71 Hex_CNN 
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-72 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-73 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-74 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-75 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-76 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-77 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-78 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-79 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-80 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-81 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-82 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-83 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-84 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-85 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-86 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-87 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-88 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-89 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-90 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-91 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-92 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-93 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-94 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-95 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-96 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-97 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-98 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-99 Hex_CNN
#sbatch --partition=beards --gres=gpu:1 --cpus-per-task=40 training_run.sh 32 1 8 IMPALA clear-inf-x-100 Hex_CNN

#experiment 5
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 IMPALA 5_hypersearch Hex_CNN 0


#expierment 7
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 AlphaZero 7_AlphaZero_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 AlphaZero 7_AlphaZero_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 A3C 7_A3C_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 A3C 7_A3C_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 APPO 7_APPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 APPO 7_APPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 DDPG 7_DDPG_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 DDPG 7_DDPG_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 DQN 7_DQN_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 DQN 7_DQN_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 ES 7_ES_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 ES 7_ES_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 IMPALA 7_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 IMPALA 7_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 PPO 7_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 PPO 7_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 SAC 7_SAC_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 SAC 7_SAC_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 TD3 7_TD3_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 TD3 7_TD3_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 PPO 7_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 PPO 7_PPO_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=9 training_run_dev.sh 1 1 8 IMPALA 7_IMPALA_parallel Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=50 training_run_dev.sh 42 1 8 IMPALA 7_IMPALA_parallel Hex_CNN 0

##eval
#sbatch training_run.sh 0 0 20 PPO Hex_CNN_GPU

#bayesian dev testing
sbatch --partition=primary --gres=gpu:0 --cpus-per-task=30 training_run.sh 22 1 8 PPO 19_Bay_hypersearch Hex_CNN 0
sbatch --partition=primary --gres=gpu:0 --cpus-per-task=30 training_run.sh 20 1 10 PPO 19_Bay_hypersearch dev_net 0
sbatch --partition=primary --gres=gpu:0 --cpus-per-task=30 training_run.sh 20 1 10 PPO 19_Bay_hypersearch dev_net_with_cnn 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=30 training_run.sh 22 1 8 AlphaZero 17_Baysian_dev Hex_CNN 0
#sbatch --partition=primary --gres=gpu:0 --cpus-per-task=30 training_run.sh 22 1 8 PPO 16_Baysian_dev FC_NET 0