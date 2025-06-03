import numpy as np
import scipy.linalg as la
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm
import qutip as qt
from qutip import Qobj, identity, sigmax, sigmay, sigmaz,ptrace
from qutip import entropy_vn


def Concurrence(H,N):
    sigma_y = np.array([[0, -1j], [1j, 0]])
    eigenvalues, eigenvectors = la.eigh(H)
    state_vector = eigenvectors[:, 0]
    state=np.conj(state_vector.T.reshape(-1, 1))
    rho=(state_vector*state)
    rho_reduced = np.zeros((4, 4), dtype=complex)
    c = 0
    counter = 0
    counterj = 2 ** (N - 2)
    rho_reduced = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        r = counterj
        for j in range(4):
            if (i <= j):
                if (i == j):
                    for w in range(2 ** (N - 2)):
                        rho_reduced[j, i] = rho_reduced[j, i] + rho[c, c]
                        
                        c = c + 1
                    rho_reduced[i, j] = rho_reduced[j, i]
                else:
                    col = counter
                    for w in range(2 ** (N - 2)):
                        rho_reduced[j, i] = rho_reduced[j, i] + rho[r, col]
                    
                        col = col + 1
                        r = r + 1
                    rho_reduced[i, j] = rho_reduced[j, i]
        counter = counter + 2 ** (N - 2)
        counterj = counterj + 2 ** (N - 2)
    rho_tilda=np.kron(sigma_y, sigma_y)@(rho_reduced@(np.kron(sigma_y, sigma_y)))
    R=sqrtm(sqrtm(rho_reduced)@rho_tilda@sqrtm(rho_reduced))
    R=np.real(R)
    eigenvalues = np.linalg.eigvals(R)
    eigenvalues=np.sort(eigenvalues)
    C=np.max([0,(eigenvalues[3]-eigenvalues[2]-eigenvalues[1]-eigenvalues[0])])
    return C

def TimeAverage_Concurrence(H_base,H_evolve,t_final,dt,N):
    sigma_y = np.array([[0, -1j], [1j, 0]])
    eigenvalues, eigenvectors = eigsh(H_base)
    state_vector_1 = qt.Qobj(eigenvectors[:, 0]) 
    time_values = np.linspace(0, t_final, dt)
    sol_1 = qt.sesolve(qt.Qobj(H_evolve), state_vector_1, time_values)
    U_1 = sol_1.states
    lambda_values = []
    concurrence_values = []
    for idx, t in enumerate(time_values):
        state_vector = U_1[idx].full().flatten()
        state=np.conj(state_vector.T.reshape(-1, 1))
        rho=(state_vector*state)
        rho_reduced = np.zeros((4, 4), dtype=complex)
        c = 0
        counter = 0
        counterj = 2 ** (N - 2)
        rho_reduced = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            r = counterj
            for j in range(4):
                if (i <= j):
                    if (i == j):
                        for w in range(2 ** (N - 2)):
                            rho_reduced[j, i] = rho_reduced[j, i] + rho[c, c]
                            
                            c = c + 1
                        rho_reduced[i, j] = rho_reduced[j, i]
                    else:
                        col = counter
                        for w in range(2 ** (N - 2)):
                            rho_reduced[j, i] = rho_reduced[j, i] + rho[r, col]
                        
                            col = col + 1
                            r = r + 1
                        rho_reduced[i, j] = rho_reduced[j, i]
            counter = counter + 2 ** (N - 2)
            counterj = counterj + 2 ** (N - 2)
        rho_tilda=np.kron(sigma_y, sigma_y)@(rho_reduced@(np.kron(sigma_y, sigma_y)))
        R=sqrtm(sqrtm(rho_reduced)@rho_tilda@sqrtm(rho_reduced))
        R=np.real(R)
        eigenvalues = np.linalg.eigvals(R)
        eigenvalues=np.sort(eigenvalues)
        C=np.max([0,(eigenvalues[3]-eigenvalues[2]-eigenvalues[1]-eigenvalues[0])])
        sum_conc+= C
        lambda_values.append(t)
    conc_avg=sum_conc/len(lambda_values)
    return(conc_avg,lambda_values)

def Fidelity(create_ising_hamiltonian,J,N):
    lambda_values=[]
    fed_values=[]
    delta = 0.01 
    minus = 0.001
    for t in range(10,200):
        h1 = t * delta
        h2 = h1 + minus
        # Define the Hamiltonians for each transverse field value
        H_base1 = create_ising_hamiltonian(J, h1, N)
        H_base2 = create_ising_hamiltonian(J, h2, N)
        
        # Get the ground state of the first Hamiltonian
        eigenvalues1, eigenvectors1 = la.eigh(H_base1)
        state_vector_1 = eigenvectors1[:, 0]

        # Get the ground state of the second Hamiltonian
        eigenvalues2, eigenvectors2 = la.eigh(H_base2)
        state_vector_2 = eigenvectors2[:, 0]

        
        
        # Calculate overlaps (fidelity)
        fed = np.abs(np.conj(state_vector_2.T) @ state_vector_1)**2
        
        # Store results for plotting
        lambda_values.append(h1)
        fed_values.append(fed)
    return lambda_values,fed_values

def VN_Enthropy(psi,N):
    psi_1 = psi.T.reshape(-1, 1)
    rho = (psi * psi_1.conj())
    rho = qt.Qobj(rho, dims=[[2]*N, [2]*N])
    rho_c=ptrace(rho,N-1) 
    S = entropy_vn(rho_c)
    
    return S

def Magnetisation(N,pauli_operator,psi):
    for i in range(N - 1):  
        M_op = kron(kron(eye(2**i, format="csr"), pauli_operator), eye(2**(N - i - 1), format="csr"), format="csr")
    M = np.vdot(psi, M_op @ psi).real / (N)
    return M









