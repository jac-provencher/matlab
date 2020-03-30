import math

class differentialEquation:

    def __init__(self, h, N, initialCondition, fonction, solution=False):

        self.step = h
        self.maxIteration = N
        self.t0, self.y0 = initialCondition
        self.fonction = fonction

    def __str__(self):
        values = zip(solver.eulerExplicite(), solver.eulerModifie(), solver.pointMilieu(), solver.RK4())
        print("t       eulerExplicite       eulerModifie        pointMilieu            RK4")
        return '\n'.join(
        f"{value1[0]:.1f}     {value1[1]:.8e}      {value2[1]:.8e}     {value3[1]:.8e}     {value4[1]:.8e}"
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
        values = [(tn ,yn)]
        yprime = lambda t, y: y + self.step*self.fonction(t, y)
        for n in range(self.maxIteration):
            yn += 0.5*self.step*(self.fonction(tn, yn) + self.fonction(tn + 0.1, yprime(tn, yn)))
            tn += self.step
            values.append((tn, yn))

        return values

    def pointMilieu(self):
        tn, yn = self.t0, self.y0
        values = [(tn ,yn)]
        for n in range(self.maxIteration):
            k = self.step*self.fonction(tn, yn)
            yn += self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k)
            tn += self.step
            values.append((tn, yn))

        return values

    def RK4(self):
        tn, yn = self.t0, self.y0
        values = [(tn ,yn)]
        for n in range(self.maxIteration):
            k1 = self.step*self.fonction(tn, yn)
            k2 = self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k1)
            k3 = self.step*self.fonction(tn + 0.5*self.step, yn + 0.5*k2)
            k4 = self.step*self.fonction(tn + self.step, yn + k3)
            yn += (k1 + 2*k2 + 2*k3 + k4)/6
            tn += self.step
            values.append((tn, yn))

        return values

fonction = lambda t, y: y + math.exp(2*t)
solution = None
condition_initiale = (0, 2)
h = 0.02
N = 500
solver = differentialEquation(h, N, condition_initiale, fonction)
print(solver)
