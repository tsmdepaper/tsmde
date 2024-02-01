import torch
import autograd.numpy as np
import psutil

from tqdm.auto import tqdm

# inverse function with regularisation (not useful for this example as p=1)
def regularised_inv(A, h = 1e-6):
    return np.linalg.inv(A + h*np.eye(A.shape[0]))

from scipy.linalg import solve_triangular
def cholesky_inv(A, reg):
    L  = np.linalg.cholesky(A + reg*np.eye(len(A)))
    Linv = solve_triangular(L.T, np.eye(len(L)))
    return Linv @ Linv.T

# regular RBF/gaussian basis
def rbf(X, xc, sigma2 = 1):
    return np.exp(-np.linalg.norm(X - xc, 2, 1)**2/sigma2)[:, None]

# vectorised RBF/gaussian basis
def rbf2(X, xc, sigma2=1):
    sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
    sumy2 = np.reshape(np.sum(xc**2, 1), (1, -1))
    D2 = sumx2 - 2*np.matmul(X, xc.T) + sumy2
    out = np.exp(-D2/sigma2)
    return out

# vectorised derivative
def drbf2(t, tc, sigma2=1):
    k = rbf2(t, tc, sigma2)
    return -2*(t - tc.T)/sigma2 * k 

# vectorised second derivative
def d2rbf2(t, tc, sigma2=1):
    k = rbf2(t, tc, sigma2)
    Diff = (4*tc.T**2 - 8*tc.T*t - 2*sigma2 + 4*t**2)
    return (Diff/sigma2**2) * k

# g function (absolute value variation)
def g_abs(t):
    return min(0.1, -(abs(t - 0.5) - 0.5))

def dg_abs(t):
    if t <= 0.1:
        return 1
    elif t >= 0.9:
        return -1
    else:
        return 0

# g function (quadratic variation)
def g_quad(t):
    return (1-t)*t
    
def dg_quad(t):
    return 1-2*t


# NW expectations for f and phi
def E_nw_f(fX, weights, verbose=False):
    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x

    Ef = np.empty((fX.shape[0], fX.shape[2]))
    for ti in progress(range(fX.shape[0]), fX.shape[0]):
        Ef[ti, :] = (weights[ti, :][:, None, None] * fX).sum((0, 1)) / (fX.shape[1]*weights[ti,:].sum())
    return Ef

def E_nw_phi(fX, Ef, weights, verbose=False, torch_device="cuda" if torch.cuda.is_available() else "cpu"):
    T, n, d = fX.shape
    batch_size = allocate_batch_size_for_Ephi(T, n, d, torch_device == "cuda")
    batch_size = min(batch_size, T)
    if verbose:
        print(f"Allocated batch size for Ephi: {batch_size}")
    if batch_size > 1:
        if verbose:
            print("Using vectorised Ephi calculation")
        return E_nw_phi_vectorised(fX, Ef, weights, batch_size, verbose=verbose, torch_device=torch_device)
    else:
        if verbose:
            print("Using looped Ephi calculation")
        return E_nw_phi_looped(fX, Ef, weights, verbose=verbose, torch_device=torch_device)

# NW expectation (pytorch)
def E_nw(fy, Kttseq, new_dims=1):
    for i in range(new_dims):
        Kttseq = Kttseq.unsqueeze(i+2)
    num = (Kttseq * fy).sum((0, 1))
    return num/(fy.shape[1]*Kttseq.sum())

def E_nw_phi_vectorised(fX, Ef, weights, batch_size, verbose=False, torch_device="cuda" if torch.cuda.is_available() else "cpu"):
    
    T, n, d = fX.shape

    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x
    
    Ef = torch.from_numpy(Ef).to(torch_device)
    weights = torch.from_numpy(weights).to(torch_device)
    fX = torch.from_numpy(fX).to(torch_device)

    Ephis = torch.empty((T, d, d))
    num_batches = T // batch_size + T % batch_size

    phi = fX - Ef[:, None, :]
    phi_mats = phi[:, :, :, None] @ phi[:, :, None, :]

    for i in progress(range(num_batches), num_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, T)
        X = phi_mats[:, None, :, :, :] * weights[:, start:end, None, None, None]
        Ephis[start:end, :, :] = X.sum((0, 2)) / (n*weights[start:end, :].sum(1)[:, None, None])

    return Ephis.cpu().detach().numpy()

def E_nw_phi_looped(fX, Ef, weights, verbose=False, torch_device="cuda" if torch.cuda.is_available() else "cpu"):
    
    T, n, d = fX.shape

    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x
    
    Ef = torch.from_numpy(Ef).to(torch_device)
    weights = torch.from_numpy(weights).to(torch_device)
    fX = torch.from_numpy(fX).to(torch_device)

    Ephis = torch.empty((T, d, d)).to(torch_device)
    for ti in progress(range(T), T):

        # get kernel distances from tsub to all tseq
        Kttseq = weights[ti, :][:, None]

        # subset time to where Kttseq > 0
        subK    = (Kttseq != 0).flatten()
        KttseqK = Kttseq[subK]
        fXK     = fX[subK, :, :]

        # calculate Ephi with batching outer product
        phi = fXK - Ef[subK, None, :]
        phi_bigbatch = phi.reshape(-1, d)
        phi_mats = torch.bmm(phi_bigbatch[:, :, None], phi_bigbatch[:, None, :])
        phi_mats = phi_mats.reshape(len(KttseqK), n, d, d)

        Ephis[ti, :, :] = E_nw(phi_mats, KttseqK, 2)

    return Ephis.cpu().detach().numpy()


def E_nw_dd_looped(input, weights, verbose=False, torch_device="cuda" if torch.cuda.is_available() else "cpu"):
    
    T, n, d, _ = input.shape

    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x
    
    input = torch.from_numpy(input).to(torch_device)
    weights = torch.from_numpy(weights).to(torch_device)

    E_dd = torch.empty((T, d, d)).to(torch_device)
    for ti in progress(range(T), T):

        # get kernel distances from tsub to all tseq
        Kttseq = weights[ti, :][:, None]

        # subset time to where Kttseq > 0
        subK    = (Kttseq != 0).flatten()
        KttseqK = Kttseq[subK]
        inputK  = input[subK, :, :, :]

        # calculate Ephi with batching outer product
        E_dd[ti, :, :] = E_nw(inputK, KttseqK, 2)

    return E_dd.cpu().detach().numpy()


def allocate_batch_size_for_Ephi(T, n, d, cuda=False):
    
    if cuda:
        free = torch.cuda.mem_get_info()[0]
    else:
        free = psutil.virtual_memory().free

    nelem = T * T * n * d * d
     
    batch_size = int(free / (nelem * 4)) - 1
    return max(batch_size, 1)

