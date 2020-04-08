import math
import numpy as np

class EDO1:

    def __init__(self, h, N, initialCondition, fonction):

        self.step = h
        self.maxIteration = N
        self.t0, self.y0 = initialCondition
        self.fonction = fonction

    def __str__(self):
        values = zip(self.eulerExplicite(), self.eulerModifie(), self.pointMilieu(), self.RK4())
        print("t         eulerExplicite        eulerModifie          pointMilieu             RK4")
        return '\n'.join(
        f"{value1[0]:^4.2f}     {value1[1]:^12.10e}     {value2[1]:^12.10e}     {value3[1]:^12.10e}     {value4[1]:^12.10e}"
        for value1, value2, value3, value4 in values)

    def eulerExplicite(self):
        tn, yn = self.t0, self.y0
        values = [(tn, yn)]
        for n in range(self.maxIteration):
            yn += self.step*self.fonction(tn, yn)
            tn += self.step
            values.append((tn, yn))

        return values

    def eulerModifie(self):
        tn, yn = self.t0, self.y0
        values = [(tn, yn)]
        for n in range(self.maxIteration):
            yprime = yn + self.step*self.fonction(tn, yn)
            yn += 0.5*self.step*(self.fonction(tn, yn) + self.fonction(tn + self.step, yprime))
            tn += self.step
            values.append((tn, yn))

        return values

    def pointMilieu(self):
        tn, yn = self.t0, self.y0
        values = [(tn, yn)]
        for n in range(self.maxIteration):
            k = self.step*self.fonction(tn, yn)
            yn += self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k)
            tn += self.step
            values.append((tn, yn))

        return values

    def RK4(self):
        tn, yn = self.t0, self.y0
        values = [(tn, yn)]
        for n in range(self.maxIteration):
            k1 = self.step*self.fonction(tn, yn)
            k2 = self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k1)
            k3 = self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k2)
            k4 = self.step*self.fonction(tn + self.step, yn + k3)
            yn += (k1 + 2*k2 + 2*k3 + k4)/6
            tn += self.step
            values.append((tn, yn))

        return values

class EDO_System:

    def __init__(self, h, N, initialCondition, fonctions):

        self.step = h
        self.maxIteration = N
        self.t0, *self.y0 = initialCondition
        self.fonctions = fonctions

    def eulerExplicite(self):

        data = {}
        tn, yn = self.t0, self.y0
        data[0] = {tn : yn}

        for n in range(1, self.maxIteration + 1):
            data[n] = {}
            yn = yn[:] + self.step * np.array([f(tn, yn[:]) for f in self.fonctions])

            tn += self.step
            data[n][tn] = yn

        return data

    def RK4(self):

        data = {}
        tn, yn = self.t0, self.y0
        data[0] = {tn : yn}

        for n in range(1, self.maxIteration + 1):
            data[n] = {}

            k1 = np.array([self.step * f(tn, yn[:]) for f in self.fonctions])
            k2 = np.array([self.step * f(tn + 0.5*self.step, list(map(lambda c: c[1] + 0.5*c[0], zip(k1, yn[:])))) for f in self.fonctions])
            k3 = np.array([self.step * f(tn + 0.5*self.step, list(map(lambda c: c[1] + 0.5*c[0], zip(k2, yn[:])))) for f in self.fonctions])
            k4 = np.array([self.step * f(tn + 0.5*self.step, list(map(lambda c: c[1] + c[0], zip(k3, yn[:])))) for f in self.fonctions])

            yn = yn[:] + (k1 + 2*k2 + 2*k3 + k4)/6

            tn += self.step
            data[n][tn] = yn

        return data

f1 = lambda t, y: y[1]
f2 = lambda t, y: 2*y[1] - y[0]
condition_initiale = (0, 2, 1)
h = 0.1
N = 1
fonctions = [f1, f2]
solver = EDO_System(h, N, condition_initiale, fonctions)

for iteration, values in solver.RK4().items():
    for t, y in values.items():
        print(f"t{iteration} = {t:.2f}, y{iteration} = {y}")
