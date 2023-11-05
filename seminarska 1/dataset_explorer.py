import csv
import matplotlib.pyplot as plt
import numpy as np

with open("dataset.csv") as file:
    reader = csv.reader(file)

    for row in reader:
        print(', '.join(row))

        xs = np.fromstring(row[1][1:-1], sep=", ")
        ys = np.fromstring(row[2][1:-1], sep=", ")

        plt.scatter(xs, ys)
        plt.show()