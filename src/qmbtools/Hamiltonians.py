import numpy as np
from scipy.sparse import csr_matrix, eye, kron

def Three_spin_hamiltonian(h, J3, J, N):
    sigma_x = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
    sigma_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))
    H = csr_matrix((2**N, 2**N), dtype=np.complex64)
    id2 = eye(2, dtype=np.complex64, format='csr')

    def add_term(coeff, sites, ops):
        nonlocal H
        term = eye(1, dtype=np.complex64, format='csr')
        for j in range(N):
            op = ops[sites.index(j)] if j in sites else id2
            term = kron(term, op, format='csr')
        H += coeff * term

    for i in range(N):
        add_term(-h, [i], [sigma_z])
    for i in range(N):
        add_term(-J, [i, (i+1)%N], [sigma_x, sigma_x])
    for i in range(N):
        add_term(-J3, [(i-1)%N, i, (i+1)%N], [sigma_x, sigma_z, sigma_x])

    return H

def Ising_hamiltonian(J, h, N):
    sigma_x = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    I = eye(2, format='csr')

    H = csr_matrix((2**N, 2**N), dtype=complex)  # initialize sparse zero matrix
    
    # Transverse field term
    for i in range(N):
        left = eye(2**i, format='csr')
        right = eye(2**(N - i - 1), format='csr')
        Hz = -h * kron(kron(left, sigma_z), right, format='csr')
        H += Hz
        
    # Interaction term
    for i in range(N - 1):
        left = eye(2**i, format='csr')
        right = eye(2**(N - i - 2), format='csr')
        Hx = -J * kron(kron(kron(left, sigma_x), sigma_x), right, format='csr')
        H += Hx

    # Periodic boundary term
    H_periodic = -J * kron(sigma_x, kron(eye(2**(N-2), format='csr'), sigma_x, format='csr'), format='csr')
    H += H_periodic

    return H

def Classical_hamiltonian(J,L):
    spin = np.zeros((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            if np.random.rand() < 0.5:
                spin[i, j] = 1
            else:
                spin[i, j] = -1
            M += spin[i, j]


    # Initialize energy and magnetization
    E = 0.0
    N = L * L

    for i in range(L):
        for j in range(L):
            a = (i + 1) % L
            b = (i - 1) % L
            c = (j + 1) % L
            d = (j - 1) % L
            E -= J * spin[i, j] * (spin[a, j] + spin[b, j] + spin[i, c] + spin[i, d])
    return E
