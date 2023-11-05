import csv
import matplotlib.pyplot as plt
import numpy as np

with open("dataset.csv") as file:
    reader = csv.reader(file)

    for row in reader:
        print(', '.join(row))

        # row[1] is x, row[2] is y
        # take substring to remove brackets
        # split by comma and space
        xs = np.fromstring(row[1][1:-1], sep=", ")
        ys = np.fromstring(row[2][1:-1], sep=", ")

        plt.scatter(xs, ys)
        plt.show()