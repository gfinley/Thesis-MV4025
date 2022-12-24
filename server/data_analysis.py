import os
import csv
from matplotlib import pyplot as plt
import numpy as np
import numpy
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from matplotlib.pyplot import figure
import pandas
import json


#do a os walk for all the files in the directory
directory = "../../ray_results"

root, dirs, files = next(os.walk(directory))

#make an array of all the files in experiment 1

#print all dirs
#for folder in dirs:
#    if "11-28" in folder:
#            print(folder)
            #print all files in the folder
            
            
Experiment1 = ["PPO_atlatl_2022-11-17_08-08-27_sb_nred","PPO_atlatl_2022-11-17_08-08-27llg1m31x","PPO_atlatl_2022-11-17_08-08-2601h1uuoe"]
Experiment1_labels = ["FC Net","standard CNN","Hexagly CNN"]

Experiment2 = ["PPO_atlatl_2022-11-17_08-09-44nuq3q05z","PPO_atlatl_2022-11-17_08-09-44s01mwzok","PPO_atlatl_2022-11-17_08-09-45a2hocnwh","PPO_atlatl_2022-11-17_08-09-45cixi4ngr"]
Experiment2_labels = ["20 workers, 2 CPU per, 8 CPU head","15 workers, 2 CPU per, 8 CPU head","10 workers, 2 CPU per, 8 CPU head","5 workers, 2 CPU per, 8 CPU head"]

Experiment3 = ["PPO_atlatl_2022-11-17_08-24-08yze5mi1c","PPO_atlatl_2022-11-17_08-24-11p1fb00li","PPO_atlatl_2022-11-17_08-24-13naqrxa_2","PPO_atlatl_2022-11-17_08-24-131oxp2put"]
Experiment3_labels =  ["20 workers, 2 CPU per, 8 CPU head","15 workers, 2 CPU per, 8 CPU head","10 workers, 2 CPU per, 8 CPU head","5 workers, 2 CPU per, 8 CPU head"]

Experiment6 = ["PPO_atlatl_2022-11-17_10-43-12dyghk765","PPO_atlatl_2022-11-17_08-24-08yze5mi1c"]
Experiment6_labels = ["0 workers, 0 CPU per, 4 CPU head", "Experiment 3 reference time"]

Experiment7 = ["PPO_atlatl_2022-11-17_13-31-42seyiyjaw","PPO_atlatl_2022-11-17_10-43-12dyghk765", "PPO_atlatl_2022-11-17_08-24-08yze5mi1c", "PPO_atlatl_2022-11-17_21-03-362gkp8f_9" ]
Experiment7_labels = ["Hexagly CNN GPU","0 workers, 0 CPU per, 4 CPU head","FC_Net (experiment 3 20/2/8)", "FC Net GPU"]

#function to read the csv file return contents as a list
def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


#function to plot the reward max from a the files in an expermeint
def plot_best_mean_max(file_list,experiemnt_name,labels):
    counter = 0
    fig,ax = plt.subplots(figsize=(10,10))

    for file in file_list:
        filename = root + "/" + file + "/progress.csv"
        data = read_csv(filename)
        
        pd = pandas.read_csv(filename)
        
        #from pd get values in episode_reward_max col
        pd_data = pd['episode_reward_max']
        
        #plot the reward_max_list vs the x list
        ax.plot(pd_data,label = labels[counter])
        counter += 1
    #label the axis x = iteration_number y = reward_max
    
    ax.set_ylabel('reward_max')
    ax.set_xlabel('iteration_number')
    
    #have y axis ticks every 100
    ax.yaxis.set_ticks(numpy.arange(0, 1000, 100))
    ax.xaxis.set_major_formatter('{x:9<5.1f}')
    ax.yaxis.set_major_formatter('{x:9<5.1f}')
    
    
    
    #show the legend
    ax.legend()
    plt.savefig("data_" +experiemnt_name + "_reward_max_list.png")
    
#function to plot the reward max from a the files in an expermeint
def plot_interation_time(file_list,experiemnt_name,labels):
    counter = 0
    fig,ax = plt.subplots(figsize=(10,10))
    #make the figure size bigger
    
    for file in file_list:
        filename = root + "/" + file + "/progress.csv"
        data = read_csv(filename)
        #get the time_this_iter_s column in the csv file
        
        pd = pandas.read_csv(filename)
        
        #from pd get values in time_this_iter_s col
        pd_data = pd['time_this_iter_s']
        
        #plot the reward_max_list vs the x list
        ax.plot(pd_data,label = labels[counter])
        counter += 1
    #label the axis x = iteration_number y = reward_max
    
    ax.set_ylabel('Iteration time')
    ax.set_xlabel('iteration_number')
    
    #have y axis ticks every 100
    ax.yaxis.set_ticks(numpy.arange(0, 300, 50))
    #show ticks every 10 iterations
    ax.xaxis.set_major_formatter('{x:9<5.1f}')
    ax.yaxis.set_major_formatter('{x:9<5.1f}')
    
    
    
    #show the legend
    ax.legend()
    plt.savefig("data_" +experiemnt_name + "iteration_time.png")
  
  
def plot_scores():  
    plot_best_mean_max(Experiment1,"Experiment1",Experiment1_labels)
    plot_interation_time(Experiment1,"Experiment1",Experiment1_labels)

    plot_best_mean_max(Experiment2,"Experiment2",Experiment2_labels)
    plot_interation_time(Experiment2,"Experiment2",Experiment2_labels)

    plot_best_mean_max(Experiment3,"Experiment3",Experiment3_labels)
    plot_interation_time(Experiment3,"Experiment3",Experiment3_labels)

    plot_best_mean_max(Experiment6,"Experiment6",Experiment6_labels)
    plot_interation_time(Experiment6,"Experiment6",Experiment6_labels)

    plot_best_mean_max(Experiment7,"Experiment7",Experiment7_labels)
    plot_interation_time(Experiment7,"Experiment7",Experiment7_labels)


#get folders that match the following date
def get_folder_names(date_dir):
    folder_names = []
    for folder in os.listdir( "../../ray_results"):
        if date_dir in folder:
            folder_names.append(folder)
    return folder_names

print(get_folder_names("11-28"))

#given a directory name load progress.csv into a numpy array and return the last row of the coloumbscustom_model episode_reward_mean, time_since_restore
def get_run_summary(dir_name):
    root =  "../../ray_results"
    filename = root + "/" + dir_name + "/progress.csv"
    #load data into pandas dataframe
    data = pandas.read_csv(filename)
    
    #get last episode_reward_mean and time_since_restore
    episode_reward_mean = data['episode_reward_mean'].iloc[-1]
    time_since_restore = data['time_since_restore'].iloc[-1]
    
    #open target json file  and get the value of the key "custom model"
    json_file = root + "/" + dir_name + "/params.json"
    custom_model = json.load(open(json_file))["model"]["custom_model"]
    
    #get number of GPUs under key num_gpus
    number_gpus = json.load(open(json_file))["num_gpus"]
    worker_number_cpus = json.load(open(json_file))["num_cpus_per_worker"]
    driver_number_cpus = json.load(open(json_file))["num_cpus_for_driver"]
    number_workers = json.load(open(json_file))["num_workers"]
    
    #get total number of timesteps
    total_timesteps = data['timesteps_total'].iloc[-1]
    
    trainer = str(dir_name[0:3])
 
    
    return trainer, custom_model, number_workers, worker_number_cpus,  driver_number_cpus, number_gpus, total_timesteps, time_since_restore, episode_reward_mean

folder_names = get_folder_names("11-28")

for folder in folder_names:
    data = get_run_summary(folder)
    print(str(data)[1:-1])
    
folder_names = get_folder_names("11-29")
    
print()

for folder in folder_names:
    data = get_run_summary(folder)
    print(str(data)[1:-1])
    
folder_names = get_folder_names("11-30")
    
print()

for folder in folder_names:
    data = get_run_summary(folder)
    print( str(data)[1:-1])