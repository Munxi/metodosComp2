import sys
import numpy as np
import matplotlib.pyplot as plt
from fontTools.subset.svg import xpath

with open("Rhodium.csv") as f:
    def a():
        linea = f.readline()
        x = []
        y = []
        linea = f.readline()
        i = 0
        while linea != "":
            long, inten = linea.split(",")
            long = float(long)
            inten = float(inten)
            freq = (long*10**-12)/(3*10**8)
            x.append(freq)
            y.append(inten)
            linea = f.readline()
            if i>=1:
                Deltax = x[i-1]-x[i]
                Deltay = y[i-1]-y[i]
                dx = Deltay/Deltax
                print(dx)
            i+=1
        # plt.scatter(x,y)
        # plt.show()
    a()



