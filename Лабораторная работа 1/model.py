import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq

class Differential_problem(object):
    def __init__(self, k, q, f, *args):
        self.k = k
        self.q = q
        self.f = f
        self.x = args

class Boundary_value_problem(Differential_problem):
    def __init__(self, k, q, f, left, right, A, b, *args):
        self.left = left
        self.right = right
        self.A = A
        self.b = b
        Differential_problem.__init__(self, k, q, f, *args)

    def computational_solution_bvp(self, L):
        h = (self.right - self.left)/L

        x = [0 for i in range(L+1)] 
        u = [0 for i in range(L+1)]

        k_val = [0 for i in range(L+2)]
        f_val = [0 for i in range(L+1)]
        q_val = [0 for i in range(L+1)]

        for i in range(L+1):
            x[i] = i * h
            f_val[i] = f(x[i])
            q_val[i] = q(x[i])
        
        k_val[0] = k(x[0])
        for i in range(1, L+1):
            k_val[i] = k(x[i] - h/2)
        k_val[L+1] = k(x[L])

        s_a = [0 for i in range(L+1)]
        s_b = [0 for i in range(L+1)]
        s_c = [0 for i in range(L+1)]
        s_d = [0 for i in range(L+1)]
        
        alpha = [0 for i in range(L)]
        beta = [0 for i in range(L)]

        s_a[0] = A[0][0]
        s_b[0] = (-A[0][0]) + A[0][1] * h
        s_d[0] = b[0] * h
        alpha[0] = (-s_a[0])/s_b[0]
        beta[0] = (s_d[0])/s_b[0]

        for i in range(1, L):
            s_a[i] = k_val[i+1] 
            s_b[i] = -(k_val[i] + k_val[i+1] + q_val[i] * h**2)
            s_c[i] = k_val[i]
            s_d[i] = (-f_val[i]) * h**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_b[L] = A[1][0] + A[1][1] * h
        s_c[L] = -A[1][0]
        s_d[L] = h * b[1]

        u[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in reversed(range(1, L+1)):
            u[i-1] = alpha[i-1] * u[i] + beta[i - 1]

        t = np.arange(self.left, self.right + h, h)

        #plt.figure(figsize=(12, 10), dpi=300)
        plt.plot(t, u, lw = 1.5, color = 'purple')
        kwargs={'linestyle':'--', 'lw':0.5}
        plt.grid(True, **kwargs)
        plt.show()
        
        return [t, u]

class Model_problem(Boundary_value_problem):
    def __init__(self, k, q, f, left, right, A, b, point, *args):
        self.point = point
        Boundary_value_problem.__init__(self, k, q, f, left, right, A, b, *args)

    def function_val(self):
        data = [self.k(self.point), self.q(self.point), self.f(self.point)]
        return data

    def characteristic(self):
        data = self.function_val()
        k_ = data[0]
        q_ = data[1]
        if(q_/k_ > 0):
            lamb = [complex(math.sqrt(q_/k_), 0), complex(-math.sqrt(q_/k_), 0)]
        else:
            lamb = [complex(0, math.sqrt(-q_/k_)), complex(0, -math.sqrt(-q_/k_))]
        return lamb

    def partial_solution(self):
        data = self.function_val()
        q_ = data[1]
        f_ = data[2]
        return f_/q_

    def coefficients(self):
        lamb  = self.characteristic()[0]
        lamb = lamb.real

        z00 = A[0][0] * lamb + A[0][1]
        z01 = (-A[0][0]) * lamb + A[0][1]
        z10 = A[1][0] * lamb * np.exp(lamb) + A[1][1] * np.exp(lamb)
        z11 = (-A[1][0]) * lamb * np.exp((-1.) * lamb) + A[1][1] * np.exp((-1.) * lamb)
        matrix = np.array([[z00, z01], 
                           [z10, z11]])
        free_members = [b[0] - A[0][1] * self.partial_solution(), b[1] - A[1][1] * self.partial_solution()]
        c = np.linalg.solve(matrix, free_members)

        return c

    def analitic_solution_mp(self, L):
        c = self.coefficients()
        c[0] = c[0].real
        c[1] = c[1].real
        lamb = self.characteristic()
        lamb = lamb[0].real
        ps = self.partial_solution()

        h = 1/L
        t = np.arange(self.left, self.right + h, h)

        s = [0 for i in range(len(t))]
        for i in range(len(t)):
            s[i] = c[0] * np.exp(lamb * t[i]) + c[1] * np.exp((-1.) * lamb * t[i]) + ps
        
        #plt.figure(figsize=(12, 10), dpi=300)
        plt.plot(t, s, lw = 1.5, color = 'darkblue')
        kwargs={'linestyle':'--', 'lw':0.5}
        plt.grid(True, **kwargs)
        plt.show()

        return [t, s]
    
    def computational_solution_mp(self, L):
        data = self.function_val()
        k_ = data[0]
        q_ = data[1]
        f_ = data[2]

        h = (self.right - self.left)/L

        s_a = [0 for i in range(L+1)]
        s_b = [0 for i in range(L+1)]
        s_c = [0 for i in range(L+1)]
        s_d = [0 for i in range(L+1)]
        u = [0 for i in range(L+1)]

        alpha = [0 for i in range(L)]
        beta = [0 for i in range(L)]

        s_a[0] = k_
        s_b[0] = A[0][1] * h - k_
        s_d[0] = h * b[0]
        alpha[0] = (-s_a[0])/s_b[0]
        beta[0] = (s_d[0])/s_b[0]

        for i in range(1, L):
            s_a[i] = k_
            s_b[i] = -(2 * k_ + q_ * h**2)
            s_c[i] = k_
            s_d[i] = (-f_) * h**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_b[L] = -k_ + A[1][1] * h
        s_c[L] = k_
        s_d[L] = h * b[1]

        u[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in reversed(range(1, L+1)):
            u[i-1] = alpha[i-1] * u[i] + beta[i - 1]

        t = np.arange(self.left, self.right + h, h)
        plt.plot(t, u, lw = 1.5, color = 'red')
        kwargs={'linestyle':'--', 'lw':0.5}
        plt.grid(True, **kwargs)
        plt.show()

        return [t, u]

def k(x):
    return math.exp(x)

def q(x):
    return math.exp(x)
    
def f(x):
    return math.cos(x)

#x = 0.0

dp = Differential_problem(k, q, f)

A = np.array([[k(0.0), -1.0],
              [-k(1.0), -1.0]])

b = np.array([-1.0, 0.0])

bvp = Boundary_value_problem(k, q, f, 0.0, 1.0, A, b)

