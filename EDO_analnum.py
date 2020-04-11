import math
import numpy as np
import sympy as sp

VARIABLES = x, y, z = sp.symbols('x y z')
NMAX = 20
TOL = 10**-8

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

    def Taylor(self):
        pass

    def eulerModifie(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            yprime = yn + self.h*self.fonctions(tn, yn)
            yn = yn + 0.5*self.h*(self.fonctions(tn, yn) + self.fonctions(tn + self.h, yprime))
            tn += self.h

            data[n][tn] = yn

        return data

    def pointMilieu(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            k = self.h*self.fonctions(tn, yn)
            yn = yn + self.h*self.fonctions(tn + 0.5*self.h, yn + 0.5*k)
            tn += self.h

            data[n][tn] = yn

        return data

    def RK4(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}

            k1 = self.h*self.fonctions(tn, yn)
            k2 = self.h*self.fonctions(tn + 0.5*self.h, yn + 0.5*k1)
            k3 = self.h*self.fonctions(tn + 0.5*self.h, yn + 0.5*k2)
            k4 = self.h*self.fonctions(tn + self.h, yn + k3)

            yn = yn + (k1 + 2*k2 + 2*k3 + k4)/6
            tn += self.h

            data[n][tn] = yn

        return data

    def eulerExplicite(self):

        tn, yn = self.t0, np.array(self.y0)
        data = {0: {tn : yn}}
        for n in range(1, self.maxIteration+1):
            data[n] = {}
            yn = yn + self.h*self.fonctions(tn, yn)
            tn += self.h

            data[n][tn] = yn

        return data

f1 = lambda t, y: y[1]
f2 = lambda t, y: -30*(y[0] + y[1])
fonctions = lambda t, y: np.array([f1(t, y), f2(t, y)])
condition_initiale = (0, 0, 30)
h = 0.01
solver = differential_equation_solver(h, NMAX, condition_initiale, fonctions)
# data = solver.eulerExplicite()
# solver.displayValues(data)

def newtonZero(nMax, x0, fct):

    TOL = 10**-10
    err_machine = 10**-15
    f = sp.lambdify(x, fct)
    fprime = sp.lambdify(x, fct.diff(x))
    x1 = x0 - f(x0)/fprime(x0)
    i = 0

    while abs(x1 - x0)/(x1 + err_machine) >= TOL and i <= nMax:

        x0 = x1
        x1 = x0 - f(x0)/fprime(x0)
        i += 1

    return x1

x0 = 0
fct = sp.exp(-x) - x
# print(newtonZero(NMAX, x0, fct))

def newtonSystem(x, functions, TOL, nMax):

    J = sp.lambdify([VARIABLES], sp.Matrix([[f.diff(variable) for variable in VARIABLES] for f in functions]))
    R = sp.lambdify([VARIABLES], sp.Matrix([[f] for f in functions]))
    dx = np.zeros((3, 1))
    n = 0

    while np.linalg.norm(residu := R(x)) >= TOL and n < nMax:
        jacobien = J(x)
        dx = np.linalg.solve(jacobien, -residu)
        x = np.array(x).reshape((len(x), 1)) + dx
        print(f"ITERATION {n+1}\nJacobien:\n{jacobien}\ndx:\n{dx}\nRésidu:\n{residu}\nx:\n{x}\n")
        x = x.reshape(len(x))

        n += 1

f2 = 2*x**2 - x*y + 3*y - 11
f1 = x**2 + 2*y**2 - 22
f3 = x**z + y**(2*x) - 43**z
functions = [f1, f2, f3]
x = (1, 2, 6)
newtonSystem(x, functions, TOL, NMAX)

