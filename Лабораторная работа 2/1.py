import numpy as np
import matplotlib.pyplot as plt
import math

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
            r'\usepackage[english, russian]{babel}',
            r'\usepackage{amsmath}',
            r'\usepackage{siunitx}']

class Partial_derivatives_equation(object):
    def __init__(self, a, b, *args):
        self.a = a
        self.b = b
        self.t = args
        self.x = args

class Mixed_problem(Partial_derivatives_equation):
    def __init__(self, a, b, left_x, right_x, left_t, right_t, phi, psi, *args):
        self.left_x = left_x
        self.right_x = right_x
        self.left_t = left_t
        self.right_t = right_t
        self.phi = phi
        self.psi = psi
        Partial_derivatives_equation.__init__(self, a, b, *args)

    def computational_solution_mp(self, L, N):
        #sampling steps along the time and coordinate axes
        h = (self.right_x - self.left_x)/L
        tau = (self.right_t - self.left_t)/N

        #grids on the time and coordinate axes
        x = np.linspace(self.left_x, self.right_x, L+1)
        t = np.linspace(self.left_t, self.right_t, N+1)

        u = [0 for n in range(len(t))]
        for n in range(len(t)):
            u[n] = [0 for l in range(len(x))]

        #a and b initialization
        a_val = [0 for n in range(len(t))]
        b_val = [0 for n in range(len(t))]
        for n in range(len(t)):
            a_val[n] = [a(x[l], t[n]) for l in range(len(x))]
            b_val[n] = [b(x[l], t[n]) for l in range(len(x))]

        #conditions phi and psi initialization
        phi_val = [0 for l in range(len(x))]
        for l in range(len(x)):
            phi_val[l] = phi(x[l])

        psi_val = [0 for n in range(N+1)]
        for n in range(N+1):
            psi_val[n] = psi(t[n])

        for l in range(L+1):
            u[0][l] = phi_val[l]

        for n in range(N+1):
            u[n][0] = psi_val[n]
        
        for n in range(N):
            for l in range(1, L+1):
                u[n+1][l] = u[n][l] - tau/h*(u[n][l] - u[n][l-1])*a_val[n][l] + tau * b_val[n][l] 

        return [t, x, u]
    
    def analytical_solution_mp(self, L, N):
        #grids on the time and coordinate axes
        x = np.linspace(self.left_x, self.right_x, L+1)
        t = np.linspace(self.left_t, self.right_t, N+1)

        u = [0 for n in range(len(t))]
        for n in range(len(t)):
            u[n] = [0 for l in range(len(x))]

        #conditions phi and psi initialization
        phi_val = [0 for l in range(len(x))]
        for l in range(len(x)):
            phi_val[l] = phi(x[l])

        psi_val = [0 for n in range(len(t))]
        for n in range(len(t)):
            psi_val[n] = psi(t[n])

        for n in range(len(t)):
            for l in range(len(x)):
                if(t[n] < x[l]):
                    u[n][l] = np.exp(t[n]) - t[n] + x[l]
                else:
                    u[n][l] = np.exp(t[n]) - t[n] + x[l]
        return [t, x, u]

def a(x, t):
    return 1

def b(x, t):
    return np.exp(t)

def phi(x):
    return x + 1

def psi(t):
    return np.exp(t) - t

pde = Partial_derivatives_equation(a, b)

left_x, right_x = 0, 1
left_t, right_t = 0, 1

mp = Mixed_problem(a, b, left_x, right_x, left_t, right_t, phi, psi)
data_comp = mp.computational_solution_mp(200, 200)
u_comp = data_comp[2]

data_analytical = mp.analytical_solution_mp(200, 200)
u_analytical = data_analytical[2]

diff = [0 for n in range(len(u_comp))]
for n in range(len(u_comp)):
    diff[n] = [float("{:.4f}".format(abs(u_analytical[n][l] - u_comp[n][l]))) for l in range(len(u_comp[n]))]

q = [0 for i in range(len(diff))]

for i in range(len(diff)):
    q[i] = max(diff[i])

s = max(q)

x, t = np.meshgrid(data_comp[1], data_comp[0])

levels = np.linspace(0, s, 21)

#-- Plot ------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

cs = ax.contourf(x, t, diff, levels)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'Значение функции $u(x, t)$', fontsize=14)

ax.set_title(r'Численное решение смешанной задачи для' + '\n' + 'уравнения с частными производными гиперболического типа', va='bottom', fontsize=20)
ax.grid(True)

plt.savefig('test-3.png', dpi=300)

fig, ax = plt.subplots(figsize=(12, 12))

u_1 = u_comp[-1]
u_2 = u_analytical[-1]

fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(data_comp[1], u_1, color='red')
ax.plot(data_comp[1], u_2, color='green')
plt.savefig('diff.png', dpi=300)



        