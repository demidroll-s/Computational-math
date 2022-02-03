import numpy 

mu = 0.5
C1 = 11
delta = 0.01
eps = 10**(-6)

N = 50
L = 100
M = 100

t = np.linspace(0, C1 * mu /(2 * (2 + mu)) - delta, N+1)
tau = (C1 * mu /(2 * (2 + mu)) - delta)/N  
        
r = []
h_r = 1/L
for l in range(2*L+1):
    r.append(h_r/2 * l)

phi = np.linspace(0, np.pi/2, M+1)   
h_phi = np.pi/(2*M)

u = [0 for n in range(N+1)]
for n in range(N+1):
    u[n] = [0 for l in range(L+1)]
    for l in range(L+1):
        u[n][l] = [0 for m in range(M+1)]

for l in range(L+1):
    for m in range(M+1):
        u[0][l][m] = (r[2*l] * np.cos(phi[m]))**4/121

for n in range(N+1):
    for m in range(M+1):
        u[n][L][m] = (np.cos(phi[m]))**4/(C1 - 2*(mu + 2)/mu * t[n])**2
        u[n][0][m] = 0

for n in range(N+1):
    for l in range(L+1):
        u[n][l][M] = 0
        u[n][l][0] = (r[2*l])**4/(C1 - 2*(mu + 2)/mu * t[n])**2

for n in range(N):
    k = n
    J = k
    while (True):
        u_ = [0 for l in range(L+1)]
        for l in range(L+1):
            u_[l] = [0 for m in range(M+1)]

        for m in range(M+1):
            u_[0][m] = 0
            u_[L][m] = (np.cos(phi[m]))**4/(C1 - 2*(mu + 2)/mu * t[n+1])**2

        for m in range(1, M):
            s_a = [0 for l in range(L+1)]
            s_b = [0 for l in range(L+1)]
            s_c = [0 for l in range(L+1)]
            s_d = [0 for l in range(L+1)]
            
            alpha = [0 for l in range(L)]
            beta = [0 for l in range(L)]

            s_a[0] = 0
            s_b[0] = 1
            s_c[0] = 0
            s_d[0] = 0

            alpha[0] = -s_a[0]/s_b[0]
            beta[0] = s_d[0]/s_b[0]

            for l in range(1, L):
                s_a[l] = -tau * r[2*l+1] /(2 * h_r**2 * r[2*l]) * ((u[k][l+1][m])**mu + (u[k][l][m])**mu)
                s_c[l] = -tau * r[2*l-1] /(2 * h_r**2 * r[2*l]) * ((u[k][l][m])**mu + (u[k][l-1][m])**mu)
                s_b[l] = 1 - s_a[l] - s_c[l]
                s_d[l] = u[n][l][m]

                alpha[l] = -s_a[l]/(s_b[l] + s_c[l] * alpha[l-1])
                beta[l] = (s_d[l] - s_c[l] * beta[l-1])/(s_b[l] + s_c[l] * alpha[l-1])

            s_a[L] = 0
            s_b[L] = 1
            s_c[L] = 0
            s_d[L] = (np.cos(phi[m]))**4/(C1 - 2*(mu + 2)/mu * t[n+1])**2

            u_[L][m] = (s_d[L] - s_c[L] * beta[L-1])/(s_b[L] + s_c[L] * alpha[L-1])

            for l in reversed(range(1, L+1)):
                u_[l-1][m] = alpha[l-1] * u_[l][m] + beta[l-1]

        for l in range(1,L):
            s_a = [0 for m in range(M+1)]
            s_b = [0 for m in range(M+1)]
            s_c = [0 for m in range(M+1)]
            s_d = [0 for m in range(M+1)]
            
            alpha = [0 for m in range(M)]
            beta = [0 for m in range(M)]

            s_a[0] = 0
            s_b[0] = 1
            s_c[0] = 0
            s_d[0] = (r[2*l])**4/(C1 - 2*(mu + 2)/mu * t[n+1])**2

            alpha[0] = (-s_a[0])/s_b[0]
            beta[0] = (s_d[0])/s_b[0]

            for m in range(1, M):
                s_a[m] = (-tau)/(2 * h_phi**2 * r[2*l]) * ((u[k][l][m+1])**mu + (u[k][l][m])**mu)
                s_c[m] = (-tau)/(2 * h_phi**2 * r[2*l]) * ((u[k][l][m])**mu + (u[k][l][m-1])**mu)
                s_b[m] = 1 - s_a[m] - s_c[m]
                s_d[m] = u_[l][m]

                alpha[m] = (-s_a[m])/(s_b[m] + s_c[m] * alpha[m-1])
                beta[m] = (s_d[m] - s_c[m] * beta[m-1])/(s_b[m] + s_c[m] * alpha[m-1])

            s_a[M] = 0
            s_b[M] = 1
            s_c[M] = 0
            s_d[M] = 0

            u[k+1][l][M] = (s_d[M] - s_c[M] * beta[M-1])/(s_b[M] + s_c[M] * alpha[M-1])

            for m in reversed(range(1, M+1)):
                u[k+1][l][m-1] = alpha[m-1] * u[k+1][l][m] + beta[m-1]

        q = [0 for l in range(L+1)]
        for l in range(L+1):
            q[l] = [0 for m in range(M+1)]

        for l in range(1, L):
            for m in range(1, M):
                q[l][m] = abs((u[k+1][l][m] - u[k][l][m]) / u[k][l][m])

        max_1_q = []
        for l in range(1, L):
            max_1_q.append(max(q[l]))
        max_2_q = max(max_1_q)

        if (max_2_q >= eps) and ((k + 1) != N):
            k = k + 1
            J = k
        else:
            break

    for l in range(1, L):
        for m in range(1, M): 
            u[n+1][l][m] = u[J+1][l][m]