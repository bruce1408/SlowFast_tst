import os
filepath = "/home/chenxi/SlowFast_tst/SlowFast-master0329/SlowFast-master/tools/828_train.txt"

w = open('1101_train.txt', 'w')
writeline = ""
with open(filepath, 'r') as f:
    lines = f.readlines()
    print(len(lines))
    writeline = lines[0]
    print(lines[0])
    print(lines[1])
    w.write(writeline)
    w.write("\n")
    num = len(lines)
    for i in range(1, num):
        if lines[i] == writeline:
            pass
        else:
            writeline = lines[i]
            w.write(writeline)
