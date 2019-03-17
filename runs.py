import os

addr = "to-run.txt"
command = "python ptb-lm.py "
with open(addr) as f:
    runs = f.readlines()
    for run in runs:
        os.system(command + run)
        print("==============================")
        print("==============================")
