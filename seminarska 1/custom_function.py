
import numpy as np
import csv
import tree_representation

def arr_str(arr):
    s = "["
    for x in arr[:-1]:
        s += str(x) + ", "
    s += str(arr[-1])
    s += "]"

    return s

functions = ["sin x", "cos x", "exp x", "log x", "sqrt x", "abs x", "neg x"]

with open("custom_dataset.csv", "w") as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["name", "x", "y"])
    xs = np.linspace(1, 100, 100).astype(int)
    for f in functions:
        t = tree_representation.parsePolishNotationToTree(f)
        ys = t.evaluate(xs)
        writer.writerow([f, arr_str(xs), arr_str(ys)])
