import subprocess

command = "python server.py city-inf-5 --blueAI pass-agg --redAI in-class --exitWhenTerminal"
n_runs = 10

for i in range(n_runs):
    value = subprocess.run(command.split(" "), capture_output=True)
    score = value.stdout.decode("utf-8").split()[1]
    print(score)