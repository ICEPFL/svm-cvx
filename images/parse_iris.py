import numpy as np

def parser_():
    o = open("Iris_.txt", "w")
    with open("Iris.csv", "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")
            if "setosa" in line[-1]:
                line[-1] = -1
            else:
                line[-1] = 1
            for item in line:
                o.write("%s," % item)
            o.write("\n")
    f.close()
    o.close()
