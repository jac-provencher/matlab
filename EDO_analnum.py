import math
import numpy as np

class differential_equation_solver:

    def __init__(self, h, N, initialCondition, fonctions):

        self.h = h
        self.maxIteration = N
        self.t0, *self.y0 = initialCondition
        self.fonctions = fonctions

    def displayValues(self, data):

        for iteration, values in data.items():
            for t, y in values.items():
                print(f"{iteration:<4} |  {t:>6.2f}  |  {y}")

    def eulerExplicite(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}
            yn = yn + self.h * np.array([f(tn, yn) for f in self.fonctions])
            tn += self.h

            data[n][tn] = yn

        return data

    def eulerModifie(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            yprime = yn + self.h*np.array([f(tn, yn) for f in self.fonctions])
            yn = yn + 0.5*self.h*(np.array([f(tn, yn) for f in self.fonctions]) + np.array([f(tn + self.h, yprime) for f in self.fonctions]))
            tn += self.h

            data[n][tn] = yn

        return data

    def pointMilieu(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            k = self.h*np.array([f(tn, yn) for f in self.fonctions])
            yn = yn + self.h*np.array([f(tn + 0.5*self.h, yn + 0.5*k) for f in self.fonctions])
            tn += self.h

            data[n][tn] = yn

        return data

    def RK4(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            k1 = np.array([self.h * f(tn, yn) for f in self.fonctions])
            k2 = np.array([self.h * f(tn + 0.5*self.h, yn + 0.5*k1) for f in self.fonctions])
            k3 = np.array([self.h * f(tn + 0.5*self.h, yn + 0.5*k2) for f in self.fonctions])
            k4 = np.array([self.h * f(tn + self.h, yn + k3) for f in self.fonctions])

            yn = yn + (k1 + 2*k2 + 2*k3 + k4)/6
            tn += self.h

            data[n][tn] = yn

        return data


# Note: y_{n} = y[n-1] (exemple: y_1 == y[0] >>> True)
f1 = lambda t, y: y[1]
f2 = lambda t, y: 2*y[1] - y[0]
condition_initiale = (0, 2, 1)
h = 0.1
N = 10
fonctions = [f1, f2]
solver = differential_equation_solver(h, N, condition_initiale, fonctions)
solver.displayValues(solver.RK4())
