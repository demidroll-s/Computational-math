import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
            r'\usepackage[english, russian]{babel}',
            r'\usepackage{amsmath}',
            r'\usepackage{siunitx}']

class Heat_transfer_equation(object):
    def __init__(self, mu):
        self.mu = mu

class Mixed_boundary_problem(Heat_transfer_equation):
    def __init__(self, mu, C1):
        self.C1 = C1
        Heat_transfer_equation.__init__(self, mu)

    def analitic_solution_mbp(self, N, L, M, delta):
        t = np.linspace(0, self.C1 * self.mu /(2 * (2 + self.mu)) - delta, N+1)
        r = np.linspace(0, 1, L+1)
        phi = np.linspace(0, np.pi/2, M+1)
        tau = (self.C1 * self.mu /(2 * (2 + self.mu)) - delta)/N
        h_r = 1/L
        h_phi = np.pi/(2*M)
        
        u = [0 for n in range(len(t))]
        for n in range(len(t)):
            u[n] = [0 for l in range(len(r))]
            for l in range(len(r)):
                u[n][l] = [0 for m in range(len(phi))]

        for n in range(len(t)):
            for l in range(len(r)):
                for m in range(len(phi)):
                    a = (r[l] * np.cos(phi[m]))**(2/self.mu) * (self.C1 - 2 * (self.mu + 2)/self.mu * t[n])**(-1/self.mu)
                    u[n][l][m] = a

        return [t, r, phi, u]

MBP = Mixed_boundary_problem(0.5, 11)
data = MBP.analitic_solution_mbp(10, 20, 20, 0.1)

r = data[1]
phi = data[2]
u = data[3]
values = u[7]

max_ = [0 for l in range(len(r))]

for l in range(len(r)):
    for m in range(len(phi)):
        max_[l] = max(values[l])

azimuths = data[2]
zeniths = data[1]

r_, theta = np.meshgrid(zeniths, azimuths)

levels = np.linspace(0, max(max_), 21)

#-- Plot ------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_rmax(1)
cs = ax.contourf(theta, r_, values, levels)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'Значение функции $u(r, \varphi)$ в фиксированный момент $t$', fontsize=14)

ax.set_rlabel_position(0.0)
ax.set_title(r'Аналитическое решение смешанной задачи для' + '\n' + 'квазилинейного уравнения теплопроводности', va='bottom', fontsize=20)
ax.grid(True)

plt.savefig('test-1.png', dpi=300)

#########################
