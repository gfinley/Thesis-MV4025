import os
import time
import pathlib
import json

CONTROL_FILE_PATH_PREFIX = "./"
AGENDA_FILE_NAME = "todo.json" # Array of objects, one per line
WORKING_FILE_NAME = "working.json"
RESULT_FILE_NAME = "results.csv"
LOCK_FILE_NAME = "lock"

def pretty_json_list(list, path):
    with open(path, 'w') as file:
        file.write("[\n")
        for i in range(len(list)):
            item = list[i]
            file.write(json.dumps(item))
            if i < len(list) - 1:
                file.write(",")
            file.write("\n")
        file.write("]\n")

def get_next_agenda_item():
    # Wait for lock
    while os.path.exists(CONTROL_FILE_PATH_PREFIX+LOCK_FILE_NAME):
        time.sleep(0.01)
    # Create lock file
    pathlib.Path(CONTROL_FILE_PATH_PREFIX+LOCK_FILE_NAME).touch()
    # Read and delete todo.txt
    with open(CONTROL_FILE_PATH_PREFIX+AGENDA_FILE_NAME, 'r') as file:
        agenda = json.load(file)
    os.remove(CONTROL_FILE_PATH_PREFIX+AGENDA_FILE_NAME)
    # Exit if done
    if not agenda:
        return None
    # Get first item
    result = agenda[0]
    # Write rest back to agenda file
    if len(agenda)>1:
        pretty_json_list(agenda[1:], CONTROL_FILE_PATH_PREFIX+AGENDA_FILE_NAME)
    # Write to working file
    if not os.path.exists(CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME):
        working = []
    else:
        with open(CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME, 'r') as file:
            working = json.load(file)
            os.remove(CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME)
    working.append(result)
    pretty_json_list(working, CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME)
    # Unlock
    os.remove(CONTROL_FILE_PATH_PREFIX+LOCK_FILE_NAME)
    return result

def write_result(task, moe_d):
    # Wait for lock
    while os.path.exists(CONTROL_FILE_PATH_PREFIX+LOCK_FILE_NAME):
        time.sleep(0.01)
    # Lock
    pathlib.Path(CONTROL_FILE_PATH_PREFIX + LOCK_FILE_NAME).touch()
    # Remove current item from working.txt
    with open(CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME, 'r') as file:
        working = json.load(file)
        os.remove(CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME)
    working.remove(task)
    if working:
        pretty_json_list(working, CONTROL_FILE_PATH_PREFIX+WORKING_FILE_NAME)
    # Write to results.csv
    with open(CONTROL_FILE_PATH_PREFIX+RESULT_FILE_NAME, 'a') as file:
        for key in task:
            file.write(key+","+str(task[key])+",")
        for key in moe_d:
            file.write(key+","+str(moe_d[key])+",")
        file.write("\n")
    # Unlock
    os.remove(CONTROL_FILE_PATH_PREFIX + LOCK_FILE_NAME)

task = get_next_agenda_item()

################ EXPERIMENT-SPECIFIC CODE BEGINS ###############
import train_experiment

# Run experiment
result = train_experiment.do_experiment(
    task['id'],
    task['seed'], 
    task['deep'], 
    task['algorithm'], 
    task['normed'],
    task['residuals']
)
# Compute MOEs
moe_d = {
    "mean": result['mean'],
    "stdev": result['stdev']
}
################ EXPERIMENT-SPECIFIC CODE ENDS ###############

write_result(task, moe_d)

