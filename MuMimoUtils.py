import numpy as np
import torch


def dft_column(N, x):
    """
    A column of DFT matrix
    :param N: size of DFT matrix
    :param x:  value of the column
    :return:    a column corresponding to x
    """
    k = np.arange(N)
    f = np.exp(- 1j * 2 * np.pi * x * k)

    return f


def DFTMatrix(N, f_type='DFT'):
    """
    Return the DFT matrix F
    :param N:   Size of DFT matrix -- N x N
    :param f_type: 'DFT' or 'UniSample'
    :return:    F -- the DFT matrix
    """
    # construct the DFT matrix
    F = np.zeros((N, N), dtype=complex)
    if f_type == 'DFT':
        for n in range(N):
            x = -1 / 2 + n / N
            F[:, n] = dft_column(N, x) / np.sqrt(N)
    elif f_type == 'UniSample':
        thetas = np.linspace(-np.pi/2, np.pi/2, N)
        k = np.arange(N)
        for n in range(N):
            F[:, n] = np.exp(-1j * np.pi * np.sin(thetas[n]) * k) / np.sqrt(N)
    return F


def generate_basis(N, M, rho=1, p_type='Gaussian'):
    """ generate
    :param p_type: type of pilot, = 'Gaussian' or 'Bernoulli'
    :param N: tx antenna num
    :param M:  pilot length
    :param rho: tx power
    :return:    M x N pilot matrix A satisfying trace(A*A^T)= rho * M * N
    """
    if p_type == 'Gaussian':
        A = np.sqrt(rho/2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    elif p_type == 'Bernoulli':
        A = np.sqrt(rho) * 2 * np.random.binomial(n=1, p=0.5, size=(M, N)) - 1
    else:
        A = np.zeros((M, N))
        print('Invalid basis type! Basis A set to 0')
    return A


def steering_vector(phi, N, lamda, D):
    """
    Generate steering vectors for one AOA
    :param N: BS antenna number
    :param phi: AOA/AOD
    :param lamda: wavelength
    :param D: antenna spacing
    :return: a(phi) -- the steering vector
    """
    k = np.arange(N)
    a = 1/np.sqrt(N) * np.exp(-1j*2*np.pi*D/lamda * np.sin(phi) * k)
    return a


def generate_sparse_ch(sys, theta_grids, ind_set, lamda, D, num):
    """
    :param sys: system parameters
    :param theta_grids: AoAs
    :param ind_set: index set containing envir_num index lists, each list contains common and individual supp
    :param lamda: wavelength
    :param D: antenna spacing
    :param num: number of channel samples
    :return: sparse channel matrix H = [h1, h2, ..., h_num]
    """
    N = sys['N']
    envir_num = ind_set.shape[0]
    h = np.zeros((N, num), dtype=complex)

    for n in range(num):
        # generate one realization of channel
        envir = np.random.choice(envir_num)
        ind_n = ind_set[envir]
        phis_n = theta_grids[ind_n]  # AoDs of the n-th sample
        
        P_n = len(phis_n)
        xi = 1/np.sqrt(2) * (np.random.randn(P_n, 1) + 1j * np.random.randn(P_n, 1))    # path gains

        h_n = np.zeros(N, dtype=complex)
        for p in range(P_n):
            a_p = steering_vector(phis_n[p] + np.random.rand()*0.06-0.03, N, lamda, D)
            h_n = h_n + 1/np.sqrt(P_n) * xi[p] * a_p
        h[:, n] = h_n

    return h


def complex_decompose(h):
    """
    h_tilde = [Re(h);Im(h)]
    :param h: complex numpy matrix N x num
    :return: h_tilde 2N x num
    """
    h_real = np.real(h)
    h_imag = np.imag(h)
    h_tilde = np.concatenate((h_real, h_imag), axis=0)
    return h_tilde


def CE_err(H, H_hat):
    """

    :param H: true h of shape num x 2N
    :param H_hat: estimated shape of num x 2N
    :return: MSE in dB
    """
    diff = H - H_hat
    # err_normal = 10 * np.log10(np.power(np.linalg.norm(diff, ord='fro', axis=(1, 2)) /
    #                                     np.linalg.norm(H, ord='fro', axis=(1, 2)), 2))
    err_normal = 10*torch.log10(torch.div(torch.norm(diff, dim=1), torch.norm(H, dim=1)) ** 2)
    err_dB = torch.mean(err_normal)
    return err_dB    # torch version


def CE_clx_err(H, H_hat):
    """
    :param H: true complex h of shape N x num
    :param H_hat: estimated complex channel shape of N x num
    :return: MSE in dB
    """
    diff = H - H_hat
    # err_normal = 10 * np.log10(np.power(np.linalg.norm(diff, ord='fro', axis=(1, 2)) /
    #                                     np.linalg.norm(H, ord='fro', axis=(1, 2)), 2))
    err_normal = 10*np.log10(np.divide(np.linalg.norm(diff, axis=0), np.linalg.norm(H, axis=0)) ** 2)
    err_dB = np.mean(err_normal)
    return err_dB    # torch version


def algo_omp(k, A, y):
    """
    FUNCTION algo_omp solves y = Ax, takes the input parameters y, A, k where
    y is the output field, A is the dataset field and k is the sparsity. It
    return the solution of x and the supports
    :param k:
    :param A:
    :param y:
    :return:
    """
    xbeg = np.zeros((A.shape[1], 1), dtype=complex)
    support = []
    temp = y
    count = 1
    while count < k + 1:
        ST = np.abs(A.T.conj() @ temp)
        b = np.argmax(ST)
        support.append(b)
        B = A[:, support]
        # xfinal = (np.linalg.inv(B.T.conj()@B)) @ B.T.conj() @ y
        # xfinal = np.linalg.pinv(A[:, support]) @ y
        xfinal, _, _, _ = np.linalg.lstsq(A[:, support], y, rcond=None)
        temp = y - A[:, support] @ xfinal
        count = count + 1
    x = xbeg
    x[support] = xfinal.reshape(-1, 1)
    support = np.sort(support)

    return x, support


def find_common_support(x_tilde_hat, thres=0.5):
    """
    retun the common support from the output samples
    :param x_tilde_hat: output channel samples
    :param thres: pencentage of threshold of number of common support
    :return:
    """
    num = x_tilde_hat.shape[0]  # number of channel samples
    N = int(x_tilde_hat.shape[1] / 2)
    x_tilde_hat_real = x_tilde_hat[:, 0:N].detach().numpy()
    x_tilde_hat_imag = x_tilde_hat[:, N:].detach().numpy()
    x_hat = x_tilde_hat_real + x_tilde_hat_imag * 1j
    x_hat_abs = np.abs(x_hat)

    # estimate the support (larger than some value)
    Omega_primes = {}
    abs_thres = 0.05
    for k in range(num):
        ind_est_k = (x_hat_abs[k] > abs_thres).nonzero()
        Omega_primes[k] = ind_est_k[0]

    # C. find support update
    num_occur = np.zeros((N, 1))
    for n in range(N):
        n_count = 0
        for k in range(num):
            if n in Omega_primes[k]:
                n_count += 1
        num_occur[n] = n_count
    mask = np.squeeze(num_occur >= num * thres)
    ind_c_est = mask.nonzero()[0]

    return ind_c_est, num_occur


def add_S_to_input(r_tilde, N, stype='Constant'):
    num = r_tilde.shape[0]
    if stype == 'Constant':
        S_pre = torch.zeros((num, N)) + 0.
        inputs = torch.cat((r_tilde, S_pre), dim=1)
    elif stype == 'Gaussian':
        S_pre = torch.randn((num, N)) * 1 + 0.5
        inputs = torch.cat((r_tilde, S_pre), dim=1)
    else:
        inputs = torch.cat(r_tilde, torch.zeros((num, N)) + 0.5)
    return inputs


def recover_complex_ch(h_tilde):
    """
    h_tilde is real and dim = 2N x num
    :param h_tilde: real vectorized numpy vector
    :return: complex h of size N x num
    """
    N = int(h_tilde.shape[0]/2)
    h_real = h_tilde[0:N, :]
    h_imag = h_tilde[N:, :]

    h = h_real + 1j * h_imag
    return h


# 这个函数允许我们在请求的GPU不存在的情况下运行代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
