import os
filepath = "train_log.txt"

w = open('log.txt', 'w')
writeline = ""
with open(filepath, 'r') as f:
    lines = f.readlines()
    writeline = lines[0]
    w.write(writeline)
    w.write("\n")
    num = len(lines)
    for i in range(1, num):
        if lines[i] == writeline:
            pass
        else:
            writeline = lines[i]
            w.write(writeline)
