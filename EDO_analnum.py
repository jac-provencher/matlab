import math

class differentialEquation:

    def __init__(self, h, N, initialCondition, fonction, solution=False):

        self.step = h
        self.maxIteration = N
        self.t0, self.y0 = initialCondition
        self.fonction = fonction
        self.solution = solution if bool(solution) else None

    def __str__(self):
        values = zip(self.eulerExplicite(), self.eulerModifie(), self.pointMilieu(), self.RK4())
        print("t       eulerExplicite         eulerModifie          pointMilieu             RK4")
        return '\n'.join(
        f"{value1[0]:.1f}     {value1[1]:.10e}     {value2[1]:.10e}     {value3[1]:.10e}     {value4[1]:.10e}"
        for value1, value2, value3, value4 in values)

    def compareWithSolution(self, method):
        if not bool(self.solution):
            return "Aucune solution analytique n'a été fournie."

        methods = {'eulerExplicite': self.eulerExplicite(), 'eulerModifie': self.eulerModifie(),
        'pointMilieu': self.pointMilieu(), 'RK4': self.RK4()}

        print(f"t            {method:^14}      |y(t) - yapprox|")
        return '\n'.join(f"{t:.1f}      {y:.8e}      {abs(y - self.solution(t)):.10e}" for t, y in methods[method])

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
        for n in range(self.maxIteration):
            yprime = yn + self.step*self.fonction(tn, yn)
            yn += 0.5*self.step*(self.fonction(tn, yn) + self.fonction(tn + self.step, yprime))
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

fonction = lambda t, y: y*math.exp(t)
condition_initiale = (0, 2)
h = 0.1
N = 10
solver = differentialEquation(h, N, condition_initiale, fonction)
print(solver)
