import math
import numpy as np
import sympy as sp



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
            yn = yn + self.h*self.fonctions(tn, yn)
            tn += self.h

            data[n][tn] = yn

        return data

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


# Note: y_{n} = y[n-1] (exemple: y_1 == y[0] >>> True)
# f1 = lambda t, y: y[1]
# f2 = lambda t, y: 2*y[1] - y[0]
# f3 = lambda t, y: y[2] + y[0]/y[1] - 2
# condition_initiale = (0, 2, 1, 3)
# h = 0.1
# N = 10
# fonctions = lambda t, y: np.array([f1(t, y), f2(t, y), f3(t, y)])
# solver = differential_equation_solver(h, N, condition_initiale, fonctions)
# solver.displayValues(solver.RK4())

variables = x, y, z = sp.symbols('x y z')
f1 = 3*x - sp.cos(y*z) - 1/2
f2 = x**2 - 81*(y + 0.1)**2 + sp.sin(z) + 1.06
f3 = sp.exp(-x*y) + 20*z + (10*math.pi - 3)/3
functions = f1, f2, f3
initialApproximation = (0.1, 0.1, -0.1)

def Newton(x, functions):

    J = sp.lambdify([variables], sp.Matrix([[f.diff(variable) for variable in variables] for f in functions]))
    R = sp.lambdify([variables], sp.Matrix([[f] for f in functions]))

    for n in range(10):
        dx = np.linalg.solve(J(x), -R(x))
        print(J(x))
        print(f"-{R(x)}")
        x = np.array(x).reshape((len(x), 1)) + dx
        print(x)
        x = tuple(value for values in x for value in values)

Newton(initialApproximation, functions)
