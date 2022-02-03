import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq

class Differential_problem(object):
    def __init__(self, k, q, f, x_0, *args):
        self.k = k
        self.q = q
        self.f = f
        self.x_0 = x_0
        self.x = args

class Boundary_value_problem(Differential_problem):
    def __init__(self, k, q, f, x_0, left, right, a, b, *args):
        self.left = left
        self.right = right
        self.a = a
        self.b = b
        Differential_problem.__init__(self, k, q, f, x_0, *args)

    def computational_solution_bvp(self, L):
        h1 = (self.x_0 - self.left)/L
        h2 = (self.right - self.x_0)/L

        x = [0 for i in range(2*L+1)] 
        u = [0 for i in range(2*L+1)]

        k_val = [0 for i in range(2*L+3)]
        f_val = [0 for i in range(2*L+1)]
        q_val = [0 for i in range(2*L+1)]

        for i in range(2*L+1):
            if (i <= L):
                x[i] = i * h1
            else:
                x[i] = x[L] + (i - L) * h2
            f_val[i] = f(x[i], x_0)
            q_val[i] = q(x[i], x_0)
        
        k_val[0] = k(x[0], x_0)
        for i in range(1, L+1):
            k_val[i] = k(x[i] - h1/2, x_0)
        k_val[L+1] = k(x[L], x_0)
        for i in range(L+1, 2*L+1):
            k_val[i+1] = k(x[i] - h2/2, x_0)
        k_val[2*L+2] = k(x[2*L], x_0)

        s_a = [0 for i in range(2*L+1)]
        s_b = [0 for i in range(2*L+1)]
        s_c = [0 for i in range(2*L+1)]
        s_d = [0 for i in range(2*L+1)]
        
        alpha = [0 for i in range(2*L+1)]
        beta = [0 for i in range(2*L+1)]

        s_a[0] = 0
        s_b[0] = 1
        s_d[0] = a
        alpha[0] = (-s_a[0])/s_b[0]
        beta[0] = (s_d[0])/s_b[0]

        for i in range(1, L):
            s_a[i] = k_val[i+1] 
            s_b[i] = -(k_val[i] + k_val[i+1] + q_val[i] * h1**2)
            s_c[i] = k_val[i]
            s_d[i] = (-f_val[i]) * h1**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_a[L] = k_val[L+2]/h2
        s_b[L] = -(k_val[L+2]/h2 + k_val[L]/h1)
        s_c[L] = k_val[L]/h1
        s_d[L] = 0
        alpha[L] = (-s_a[L])/(s_b[L] + s_c[L] * alpha[L-1])
        beta[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in range(L+1, 2*L+1):
            s_a[i] = k_val[i+2] 
            s_b[i] = -(k_val[i+1] + k_val[i+2] + q_val[i] * h2**2)
            s_c[i] = k_val[i+1]
            s_d[i] = (-f_val[i]) * h2**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_b[2*L] = 1
        s_c[2*L] = 0
        s_d[2*L] = b
        
        u[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in reversed(range(1, 2*L+1)):
            u[i-1] = alpha[i-1] * u[i] + beta[i - 1]

        #plt.figure(figsize=(12, 10), dpi=300)
        plt.plot(x, u, lw = 1.5, color = 'purple')
        kwargs={'linestyle':'--', 'lw':0.5}
        plt.grid(True, **kwargs)
        plt.show()
        
        return [x, u]
        


class Model_problem(Boundary_value_problem):
    def __init__(self, k, q, f, x_0, left, right, a, b, point, *args):
        self.point = point
        Boundary_value_problem.__init__(self, k, q, f, x_0, left, right, a, b, *args)

    def function_val(self):
        eps = 1e-5
        data_left = [self.k(self.point - eps, self.x_0), self.q(self.point - eps, self.x_0), self.f(self.point - eps, self.x_0)]
        data_right = [self.k(self.point + eps, self.x_0), self.q(self.point + eps, self.x_0), self.f(self.point + eps, self.x_0)]
        return [data_left, data_right]

    def characteristic(self):
        data_left, data_right = self.function_val()
        k_1 = data_left[0]
        q_1 = data_left[1]
        k_2 = data_right[0]
        q_2 = data_right[1]
        if(q_1/k_1 > 0):
            lamb_1 = [complex(math.sqrt(q_1/k_1), 0), complex(-math.sqrt(q_1/k_1), 0)]
        else:
            lamb_1 = [complex(0, math.sqrt(-q_1/k_1)), complex(0, -math.sqrt(-q_1/k_1))]
        if(q_2/k_2 > 0):
            lamb_2 = [complex(math.sqrt(q_2/k_2), 0), complex(-math.sqrt(q_2/k_2), 0)]
        else:
            lamb_2 = [complex(0, math.sqrt(-q_2/k_2)), complex(0, -math.sqrt(-q_2/k_2))]
        return [lamb_1, lamb_2]

    def partial_solution(self):
        data_left, data_right = self.function_val()
        q_1 = data_left[1]
        f_1 = data_left[2]
        q_2 = data_right[1]
        f_2 = data_right[2]
        return [f_1/q_1, f_2/q_2] 

    def coefficients(self):
        lamb  = self.characteristic()
        lamb_1 = float(lamb[0][0].real)
        lamb_2 = float(lamb[1][0].real)
        data_left, data_right = self.function_val()
        k_1 = data_left[0]
        k_2 = data_right[0]

        C = [0 for i in range(4)]
        r = [0 for i in range(4)]
        C[0] = [1, 1, 0, 0]
        C[1] = [np.exp(lamb_1 * self.x_0), np.exp(-lamb_1 * self.x_0), -np.exp(lamb_2 * self.x_0), -np.exp(-lamb_2 * self.x_0)]
        C[2] = [k_1 * np.exp(lamb_1 * self.x_0) * lamb_1, (-k_1) * np.exp(-lamb_1 * self.x_0) * lamb_1, (-k_2) * np.exp(lamb_2 * self.x_0) * lamb_2, k_2 * np.exp(-lamb_2 * self.x_0) * lamb_2]
        C[3] = [0, 0, np.exp(lamb_2), np.exp(-lamb_2)]

        r[0] = self.a - self.partial_solution()[0]
        r[1] = self.partial_solution()[1] - self.partial_solution()[0]
        r[2] = 0
        r[3] = self.b - self.partial_solution()[1]

        return [C, r]


    def analitic_solution_mp(self, L):
        [coeff, row] = self.coefficients()
        C = np.linalg.solve(coeff, row)

        lamb  = self.characteristic()
        lamb_1 = float(lamb[0][0].real)
        lamb_2 = float(lamb[1][0].real)
        ps = self.partial_solution()
        ps_1 = ps[0]
        ps_2 = ps[1]

        h1 = (self.x_0 - self.left)/L
        h2 = (self.right - self.x_0)/L

        t = [0 for i in range(2*L+1)]

        for i in range(2*L+1):
            if (i <= L):
                t[i] = i * h1
            else:
                t[i] = t[L] + (i - L) * h2
        
        s = [0 for i in range(len(t))]

        for i in range(len(t)):
            if(t[i] <= self.x_0):
                s[i] = C[0] * np.exp(lamb_1 * t[i]) + C[1] * np.exp(-lamb_1 * t[i]) + ps_1
            else:
                s[i] = C[2] * np.exp(lamb_2 * t[i]) + C[3] * np.exp(-lamb_2 * t[i]) + ps_2
        
        #plt.figure(figsize=(12, 10), dpi=300)
        plt.plot(t, s, lw = 1.5, color = 'darkblue')
        kwargs={'linestyle':'--', 'lw':0.5}
        plt.grid(True, **kwargs)
        plt.show()

        return [t, s]

    def computational_solution_mp(self, L):
        h1 = (self.x_0 - self.left)/L
        h2 = (self.right - self.x_0)/L

        x = [0 for i in range(2*L+1)] 
        u = [0 for i in range(2*L+1)]

        eps = 1e-5
        k_left = k(self.point - eps, x_0)
        q_left = q(self.point - eps, x_0)
        f_left = f(self.point - eps, x_0)

        k_right = k(self.point + eps, x_0)
        q_right = q(self.point + eps, x_0)
        f_right = f(self.point + eps, x_0)

        for i in range(2*L+1):
            if (i <= L):
                x[i] = i * h1
            else:
                x[i] = x[L] + (i - L) * h2
        
        s_a = [0 for i in range(2*L+1)]
        s_b = [0 for i in range(2*L+1)]
        s_c = [0 for i in range(2*L+1)]
        s_d = [0 for i in range(2*L+1)]
        
        alpha = [0 for i in range(2*L+1)]
        beta = [0 for i in range(2*L+1)]

        s_a[0] = 0
        s_b[0] = 1
        s_d[0] = a
        alpha[0] = (-s_a[0])/s_b[0]
        beta[0] = (s_d[0])/s_b[0]

        for i in range(1, L):
            s_a[i] = k_left
            s_b[i] = -(2 * k_left + q_left * h1**2)
            s_c[i] = k_left
            s_d[i] = (-f_left) * h1**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_a[L] = k_right/h2
        s_b[L] = -(k_right/h2 + k_left/h1)
        s_c[L] = k_left/h1
        s_d[L] = 0
        alpha[L] = (-s_a[L])/(s_b[L] + s_c[L] * alpha[L-1])
        beta[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in range(L+1, 2*L+1):
            s_a[i] = k_right
            s_b[i] = -(2 * k_right + q_right * h2**2)
            s_c[i] = k_right
            s_d[i] = (-f_right) * h2**2
            alpha[i] = (-s_a[i])/(s_b[i] + s_c[i] * alpha[i-1])
            beta[i] = (s_d[i] - s_c[i] * beta[i-1])/(s_b[i] + s_c[i] * alpha[i-1])

        s_b[2*L] = 1
        s_c[2*L] = 0
        s_d[2*L] = b
        
        u[L] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

        for i in reversed(range(1, 2*L+1)):
            u[i-1] = alpha[i-1] * u[i] + beta[i - 1]
        
        return [x, u]

def k(x, x_0):
    if (x < x_0):
        return 1
    elif (x > x_0):
        return np.exp(np.sin(x))
    else:
        return (np.exp(np.sin(x)) + 1)/2

def q(x, x_0):
    if (x < x_0):
        return 1
    else:
        return 2
    
def f(x, x_0):
    if (x < x_0):
        return np.exp(x)
    else:
        return np.exp(x)

x_0 = 1/np.sqrt(2)

dp = Differential_problem(k, q, f, x_0)

print(type(dp.x_0))

a = 1
b = 0

L = 40

bvp = Boundary_value_problem(k, q, f, x_0, 0.0, 1.0, a, b)
mp = Model_problem(k, q, f, x_0, 0.0, 1.0, a, b, x_0) 
mp.analitic_solution_mp(L)

"""
plt.figure()
plt.plot(x1, u1, lw = 1.5, color = 'purple')
plt.plot(x2, u2, lw = 1.5, color = 'darkblue')
kwargs={'linestyle':'--', 'lw':0.5}
plt.grid(True, **kwargs)
plt.show()
"""

