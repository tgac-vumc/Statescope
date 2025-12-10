import os
from termios import B0

# Disable CUDA graphs in Inductor — safer under multi-thread/multi-proc joblib
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

TMP = os.environ.get("JOBLIB_TEMP_FOLDER", "/dev/shm")

# New imports
import torch
import torch.special
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse
import dill
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import itertools
import time
import os, math
import warnings
from timeit import default_timer as timer
from functools import partial
import importlib
from contextlib import contextmanager
from contextlib import nullcontext
import numpy as np
import contextlib


def ExpF_C(Beta):
    # Compute normalized Beta across the second axis (cells).
    return Beta / torch.sum(Beta, dim=1, keepdim=True)

# @torch_prof
def ExpQ_C(Nu, Beta, Omega):
    # Expected value of F (Nsample by Ncell)
    ExpB = ExpF_C(Beta)

    # Element-wise exponential and broadcasting of Nu and Omega
    out = torch.sum(ExpB.unsqueeze(1) * torch.exp(Nu + 0.5 * Omega**2), dim=-1)
    return out.T


def VarQ_C(
    Nu, Beta, Omega, Ngene, Ncell, Nsample,
    *, chunk_G: int | None = None, cap_bytes: int = 64 * 1024 * 1024
):
    """
    Chunked Torch version of original VarQ (matches legacy math).
    Returns: (G, S)
    """
    eps = 1e-12
    S, G, C = Nu.shape
    assert (S, G, C) == (Nsample, Ngene, Ncell)
    assert Beta.shape == (S, C) and Omega.shape == (G, C)

    device, dtype = Nu.device, Nu.dtype

    # --- Dirichlet moments (once) ---
    B0     = Beta.sum(dim=1).clamp_min(eps)                          # (S,)
    Btilda = Beta / B0.unsqueeze(1)                                   # (S,C)
    VarB   = Btilda * (1.0 - Btilda) / (B0 + 1.0).unsqueeze(1)        # (S,C)
    CovB   = -(Btilda.unsqueeze(2) * Btilda.unsqueeze(1)) / (1.0 + B0).unsqueeze(1).unsqueeze(2)  # (S,C,C)

    out = torch.empty((G, S), dtype=dtype, device=device)

    # --- choose chunk size over genes ---
    if chunk_G is None:
        elt = Nu.element_size()
        denom = max(1, 4 * S * C * elt)      # rough live bytes per gene in chunk
        chunk_G = max(1, cap_bytes // denom)

    def _bmm_sc_sgc(weights_sc, mat_sgc):
        # (S,C) · (S,g,C) -> (S,g)
        return torch.bmm(weights_sc.unsqueeze(1), mat_sgc.transpose(1, 2).contiguous()).squeeze(1)

    for g0 in range(0, G, chunk_G):
        g1 = min(G, g0 + chunk_G)

        Nu_s = Nu[:, g0:g1, :]                    # (S, g, C)
        Om2  = Omega[g0:g1, :].square()           # (g, C)

        # Lognormal moments
        v     = torch.exp(Nu_s + 0.5 * Om2.unsqueeze(0))      # (S,g,C)   = E[X]
        v2    = torch.exp(2.0 * Nu_s + Om2.unsqueeze(0))      # (S,g,C)   = (E[X])^2
        expX2 = v2 * torch.exp(Om2).unsqueeze(0)              # (S,g,C)   = E[X^2]

        # VarTerm over c
        W1 = VarB + Btilda.square()                            # (S,C)
        W2 = Btilda.square()                                   # (S,C)
        VarTerm_Sg = _bmm_sc_sgc(W1, expX2) - _bmm_sc_sgc(W2, v2)   # (S,g)

        # CovTerm = sum_{l!=k} v_l v_k * CovB_{l,k}
        quad_all = torch.einsum('sgc,sck,sgk->sg', v, CovB, v)                  # (S,g)
        diagB    = torch.diagonal(CovB, dim1=1, dim2=2)                         # (S,C)
        diag_sub = torch.einsum('sgc,sc,sgc->sg', v, diagB, v)                  # (S,g)
        CovTerm_Sg = quad_all - diag_sub                                        # (S,g)

        out[g0:g1, :] = (VarTerm_Sg + CovTerm_Sg).transpose(0, 1)               # (g,S)

    return out  # (G,S)

# @torch_prof
def Estep_PY_C(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_C(Nu, Beta, Omega)

    a = Var / Exp / Exp

    return torch.sum(-0.5 / torch.square(SigmaY) * (a + torch.square((Y - torch.log(Exp)) - 0.5 * a)))


# @torch_prof
def Estep_PX_C(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = torch.sum(Nu, 0) / Nsample  # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5 * Nsample  # Posterior Alpha

    ExpBetaN = (
        Beta0
        + (Nsample - 1) / 2 * torch.square(Omega)
        + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (torch.square(Omega) / Nsample + torch.square(NuExp - Mu0))
    )

    # Vectorize the for loop
    ExpBetaN = ExpBetaN + 0.5 * torch.sum(torch.square(Nu - NuExp.unsqueeze(0)), dim=0)

    return torch.sum(-AlphaN * torch.log(ExpBetaN))


# @torch_prof
def grad_Nu_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # Shapes shorthand
    S, G, C = Nsample, Ngene, Ncell
    assert Nu.shape == (S, G, C) and Omega.shape == (G, C) and Beta.shape == (S, C)

    EPS = 1e-12

    # ---------- 1) grad_PX ----------
    AlphaN = Alpha0 + S * 0.5
    NuExp = torch.sum(Nu, dim=0) / S  # (G,C)

    # Accumulator for ExpBetaN (vectorized)
    ExpBetaN = Beta0 + (S - 1) / 2 * Omega.square()
    ExpBetaN += Kappa0 * S / (2 * (Kappa0 + S)) * (Omega.square() / S + (NuExp - Mu0).square())
    ExpBetaN += 0.5 * torch.sum((Nu - NuExp).square(), dim=0)  # (G,C)

    Diff_mean = torch.mean(Nu - NuExp, dim=0)  # (G,C)
    Nominator = Nu - NuExp - Diff_mean + (Kappa0 / (Kappa0 + S)) * (NuExp - Mu0)  # (S,G,C)

    grad_PX = -AlphaN * Nominator / ExpBetaN  # (S,G,C)

    # ---------- 2) Dirichlet bits: E[F], Var[F] ----------
    B0 = Beta.sum(dim=1).clamp_min(EPS)  # (S,)
    mu = ExpF_C(Beta)  # (S,C)
    mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(EPS)  # ensure rows sum to 1
    VarB = mu * (1.0 - mu) / (B0 + 1.0).unsqueeze(1)  # (S,C)

    # ---------- 3) Moments of X (log-normal) ----------
    # v = E[X], exp2 = E[X^2], v2 = (E[X])^2
    v = torch.exp(Nu + 0.5 * Omega.unsqueeze(0).square())  # (S,G,C)
    exp2 = torch.exp(2.0 * Nu + 2.0 * Omega.unsqueeze(0).square())  # (S,G,C)
    v2 = torch.exp(2.0 * Nu + Omega.unsqueeze(0).square())  # (S,G,C) == (E[X])^2

    # Helper: (S,C) x (S,G,C) -> (S,G) via bmm over C
    def _bmm_sum_over_c(W_sc, M_sgc):
        return torch.bmm(W_sc.unsqueeze(1), M_sgc.transpose(1, 2).contiguous()).squeeze(1)  # (S,G)

    # ---------- 4) ExpQ and VarQ (as in your pipeline) ----------
    Exp = ExpQ_C(Nu, Beta, Omega)  # (G,S)
    Var = VarQ_C(Nu, Beta, Omega, G, C, S)  # (G,S)

    # ---------- 5) g_Exp wrt Nu ----------
    # d E[X] / d Nu = E[X]; multiply by mu to match weighting
    g_Exp = (v * mu.unsqueeze(1)).permute(1, 2, 0)  # (G,C,S)

    # ---------- 6) g_Var wrt Nu without 4-D tensors ----------
    # VarTerm derivative:
    # d/dNu:  sum_c [ E[X^2]*(VarF+mu^2) - (E[X])^2 * mu^2 ]
    #  ->     2*E[X^2]*(VarF+mu^2)      - 2*(E[X])^2 * mu^2
    first_term = 2.0 * exp2 * (VarB + mu.square()).unsqueeze(1)  # (S,G,C)
    second_term = 2.0 * v2 * (mu.square()).unsqueeze(1)  # (S,G,C)
    g_Var_main = (first_term - second_term).permute(1, 2, 0)  # (G,C,S)

    # Covariance contribution (closed form for the double loop):
    # CovTerm_{s,g,l} = 2 * v_{s,g,l} * sum_{k!=l} v_{s,g,k} * CovB_{s,l,k}
    # with CovB_{s,l,k} = -mu_{s,l} mu_{s,k} / (1+B0_s)
    # => CovTerm_{s,g,l} = -2 * v_l * mu_l/(1+B0) * ( sum_c mu_c v_c - mu_l v_l )
    sum_mu_v = _bmm_sum_over_c(mu, v)  # (S,G)
    factor = -2.0 * (mu / (1.0 + B0).unsqueeze(1))  # (S,C)
    CovTerm_SGC = factor.unsqueeze(1) * v * (sum_mu_v.unsqueeze(2) - (mu.unsqueeze(1) * v))  # (S,G,C)
    CovTerm_GCS = CovTerm_SGC.permute(1, 2, 0)  # (G,C,S)

    g_Var = g_Var_main + CovTerm_GCS  # (G,C,S)

    # ---------- 7) Assemble a, b, grad_PY  ----------
    Exp_g1S = Exp.unsqueeze(1)  # (G,1,S)

    # a[g,c,s] = ( g_Var - 2 * g_Exp * Var/Exp ) / Exp^2
    a = (g_Var - 2.0 * g_Exp * (Var.unsqueeze(1) / Exp_g1S)) / (Exp_g1S.square())  # (G,C,S)

    Diff = Y - torch.log(Exp) - Var / (2.0 * Exp.square())  # (G,S)

    # b[g,c,s] = -Diff * ( 2*g_Exp/Exp + a )
    b = -Diff.unsqueeze(1) * (2.0 * g_Exp / Exp_g1S + a)  # (G,C,S)

    # grad_PY[g,c] per sample: sum_s  -0.5 / SigmaY^2 * (a + b)
    scaling = 0.5 / SigmaY.square().unsqueeze(1)  # (G,1,S)
    sum_ab = a + b  # (G,C,S)
    grad_PY = -(scaling * sum_ab).permute(2, 0, 1)  # (S,G,C)

    # ---------- 8) Final combine ----------
    return grad_PX * (1.0 / weight) + grad_PY


# @torch_prof
def grad_Omega_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    """
    Gradient wrt Omega with identical math to your original, but no 4-D tensors or (l,k) loops.
    Shapes (same as your code):
      Nu:    (Nsample, Ngene, Ncell)
      Omega: (Ngene,  Ncell)
      Beta:  (Nsample, Ncell)
      Y, SigmaY, ExpQ, VarQ: (Ngene, Nsample)
    Returns:
      (Ngene, Ncell)
    """
    S, G, C = Nsample, Ngene, Ncell
    assert Nu.shape == (S, G, C) and Omega.shape == (G, C) and Beta.shape == (S, C)
    EPS = 1e-12

    # -------- 1) grad_PX (same as original) --------
    AlphaN = Alpha0 + S * 0.5
    NuExp = torch.sum(Nu, dim=0) / S  # (G,C)
    ExpBetaN = (
        Beta0
        + (S - 1) / 2 * Omega.square()
        + Kappa0 * S / (2 * (Kappa0 + S)) * (Omega.square() / S + (NuExp - Mu0).square())
    )
    ExpBetaN = ExpBetaN + 0.5 * torch.sum((Nu - NuExp).square(), dim=0)  # (G,C)

    Nominator = -AlphaN * (S - 1) * Omega + (Kappa0 / (Kappa0 + S)) * Omega  # (G,C)
    grad_PX = Nominator / ExpBetaN  # (G,C)

    # -------- 2) Dirichlet pieces (E[F] and Var[F]) --------
    B0 = Beta.sum(dim=1).clamp_min(EPS)  # (S,)
    mu = ExpF_C(Beta)  # (S,C)
    mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(EPS)  # ensure rows sum to 1
    VarB = mu * (1.0 - mu) / (B0 + 1.0).unsqueeze(1)  # (S,C)

    # -------- 3) Moments of X (log-normal) --------
    # v = E[X], exp2 = E[X^2], v2 = (E[X])^2
    v = torch.exp(Nu + 0.5 * Omega.unsqueeze(0).square())  # (S,G,C)
    exp2 = torch.exp(2.0 * Nu + 2.0 * Omega.unsqueeze(0).square())  # (S,G,C)
    v2 = torch.exp(2.0 * Nu + Omega.unsqueeze(0).square())  # (S,G,C) == (E[X])^2

    # Helper: (S,C) x (S,G,C) -> (S,G) via batched GEMM over C
    def _bmm_sum_over_c(W_sc, M_sgc):
        return torch.bmm(W_sc.unsqueeze(1), M_sgc.transpose(1, 2).contiguous()).squeeze(1)  # (S,G)

    # -------- 4) Expectation/variance of Q  --------
    Exp = ExpQ_C(Nu, Beta, Omega)  # (G,S)
    Var = VarQ_C(Nu, Beta, Omega, G, C, S)  # (G,S)

    # -------- 5) g_Exp term (vectorized) --------
    # g_Exp[g,c,s] = E[X]_{s,g,c} * mu_{s,c} * Omega_{g,c}
    g_Exp = (v * mu.unsqueeze(1) * Omega.unsqueeze(0)).permute(1, 2, 0)  # (G,C,S)

    # -------- 6) g_Var = derivative of Var[Q] wrt Omega --------
    # First term (from E[X^2]) and second term (diagonal of CovX)
    # first_term(s,g,c)   = 2 * E[X^2] * (VarF + mu^2)
    # second_term(s,g,c)  = 2 * (E[X])^2 * mu^2
    first_term = 2.0 * exp2 * (VarB + mu.square()).unsqueeze(1)  # (S,G,C)
    second_term = 2.0 * v2 * (mu.square()).unsqueeze(1)  # (S,G,C)

    # Multiply by Omega and permute to (G,C,S)
    first_term_permuted = first_term.permute(1, 2, 0)  # (G,C,S)
    second_term_permuted = second_term.permute(1, 2, 0)  # (G,C,S)
    g_Var_main = (
        2.0 * Omega.unsqueeze(2) * first_term_permuted - 1.0 * Omega.unsqueeze(2) * second_term_permuted
    )  # (G,C,S)

    # -------- 7) Covariance contribution with no 4-D / no loops --------
    # Closed form for the loop:
    # CovTerm_{s,g,l} = - [ 2 * Omega_{g,l} * mu_{s,l} * v_{s,g,l} / (1+B0_s) ] * (sum_c mu_{s,c} v_{s,g,c} - mu_{s,l} v_{s,g,l})
    sum_mu_v = _bmm_sum_over_c(mu, v)  # (S,G)
    factor = -2.0 * (mu / (1.0 + B0).unsqueeze(1))  # (S,C)
    # Broadcast over (S,G,C)
    CovTerm_SGC = factor.unsqueeze(1) * v * (sum_mu_v.unsqueeze(2) - (mu.unsqueeze(1) * v))  # (S,G,C)
    CovTerm_GCS = Omega.unsqueeze(2) * CovTerm_SGC.permute(1, 2, 0)  # (G,C,S)

    g_Var = g_Var_main + CovTerm_GCS  # (G,C,S)

    # -------- 8) Combine to form 'a' and 'b' as in your code --------
    # a[g,c,s] = ( g_Var[g,c,s] - 2 * g_Exp[g,c,s] * Var[g,s] / Exp[g,s] ) / (Exp[g,s]^2)
    Exp_g1S = Exp.unsqueeze(1)  # (G,1,S)
    a = (g_Var - 2.0 * g_Exp * (Var.unsqueeze(1) / Exp_g1S)) / (Exp_g1S.square())  # (G,C,S)

    # Diff[g,s] = Y - log(Exp) - Var/(2*Exp^2)
    Diff = Y - torch.log(Exp) - Var / (2.0 * Exp.square())  # (G,S)

    # b[g,c,s] = -Diff[g,s] * ( 2*g_Exp[g,c,s]/Exp[g,s] + a[g,c,s] )
    b = -Diff.unsqueeze(1) * (2.0 * g_Exp / Exp_g1S + a)  # (G,C,S)

    # grad_PY[g,c] = sum_s -0.5 / SigmaY[g,s]^2 * (a[g,c,s] + b[g,c,s])
    grad_PY = torch.sum(-0.5 / SigmaY.square().unsqueeze(1) * (a + b), dim=2)  # (G,C)

    # -------- 9) Q(X) term --------
    grad_QX = -S / Omega  # (G,C)

    # -------- 10) Final combine  --------
    return grad_PX * (1.0 / weight) + grad_PY - grad_QX * (1.0 / weight)


# @torch_prof
def g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    # Compute exp(Nu) element-wise
    ExpX = torch.exp(Nu)  # Shape: (Nsample, Ngene, Ncell)
    # Apply the element-wise multiplication with Omega, which has shape (Ngene, Ncell)
    ExpX = ExpX * torch.exp(0.5 * torch.square(Omega)).unsqueeze(0)

    # B0mat computation (element-wise division)
    B0mat = Beta / torch.square(B0.unsqueeze(1)).clamp_min(1e-12)  # Shape: (Nsample, Ncell)

    tExpX = ExpX.transpose(1, 2)  # Shape: (Nsample, Ncell, Ngene)

    # Perform dot product of B0mat and tExpX
    B0mat = torch.matmul(B0mat.unsqueeze(1), tExpX).squeeze(1)  # Shape: (Nsample, 1, Ngene)

    g_Exp = torch.empty((Nsample, Ncell, Ngene), dtype=Nu.dtype, device=Nu.device)

# Explicit loop to match legacy semantics
    for c in range(Ncell):
        g_Exp[:, c, :] = (ExpX[:, :, c] / B0.unsqueeze(1)) - B0mat

    return g_Exp


# @torch_prof

def g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    # Broadcasting B0
    B0Rep = B0.unsqueeze(1)

    # Computing aa and aaNotT
    aa = (B0Rep - Beta) * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa / (torch.pow(B0Rep, 3) * torch.square(B0Rep + 1))
    aa += 2 * Beta * (B0Rep - Beta) / torch.pow(B0Rep, 3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (torch.pow(B0Rep, 3) * torch.square(B0Rep + 1))
    aaNotT += 2 * Beta * (-Beta) / torch.pow(B0Rep, 3)

    # ExpX2 computation
    ExpX2 = torch.exp(2 * Nu + 2 * torch.square(Omega))

    # g_Var computation (initial step)
    g_Var = torch.transpose(ExpX2, 1, 2) * aa.unsqueeze(2)

    # # Add aaNotT contributions excluding the diagonal, using the 'Omega' pattern
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                 g_Var[:, i, :] += ExpX2[:, :, j] * aaNotT[:, j].unsqueeze(-1)

    # Element-wise Beta computations
    B_B02 = Beta / torch.square(B0Rep)
    B0B0_1 = B0Rep * (B0Rep + 1)
    B2_B03 = torch.square(Beta) / torch.pow(B0Rep, 3)

    # ExpX computation
    ExpX = torch.exp(2 * Nu + torch.square(Omega))

    # Subtract B_B02 terms
    g_Var -= 2 * torch.transpose(ExpX, 1, 2) * B_B02.unsqueeze(2)

    Dot = torch.sum(B2_B03.unsqueeze(1) * ExpX, dim=2)

    # Add Dot contributions
    g_Var += 2 * Dot.unsqueeze(1)

    # CovX computation
    ExpX = torch.exp(Nu + 0.5 * torch.square(Omega))
    CovX = torch.einsum('sil,sik->silk', ExpX, ExpX)

    # gradCovB computation
    B03_2_B03_B0_1 = (3 * B0 + 2) / torch.pow(B0, 3) / torch.square(B0 + 1)
    gradCovB = torch.einsum('sl,sk->slk', Beta, Beta) * B03_2_B03_B0_1.unsqueeze(1).unsqueeze(1)

    # CovTerm1 and CovTerm2 computation using the 'Omega' pattern
    CovTerm1 = torch.zeros((Nsample, Ncell, Ncell, Ngene), dtype=Nu.dtype, device=Nu.device)
    CovTerm2 = torch.zeros((Nsample, Ncell, Ncell, Ngene), dtype=Nu.dtype, device=Nu.device)

    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / torch.square(B0B0_1)
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                CovTerm1[:, l, k, :] = gradCovB[:, l, k].unsqueeze(-1) * CovX[:, :, l, k]
                CovTerm2[:, l, k, :] = B_B0_1_B0B0_1[:, l].unsqueeze(-1) * CovX[:, :, l, k]

    # Final accumulation for g_Var
    g_Var += torch.sum(CovTerm1, dim=[1, 2]).unsqueeze(1)
    g_Var -= 2 * torch.sum(CovTerm2, dim=1)
    return g_Var


def g_PY_Beta_C(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):
    eps = 1e-12
    device = Beta.device
    dtype  = Beta.dtype

    # Ensure tensors on the same device/dtype
    Y       = Y.to(dtype=dtype, device=device)
    SigmaY  = SigmaY.to(dtype=dtype, device=device)

    # Expectations & their gradients (must be your existing Torch fns)
    Exp = ExpQ_C(Nu, Beta, Omega).to(dtype=dtype, device=device)  # (G, S)
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample).to(dtype=dtype, device=device)  # (G, S)

    Exp = torch.clamp(Exp, min=eps)  # avoid log(0)
    Var = torch.clamp(Var, min=eps)

    g_Exp = g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample).to(dtype=dtype, device=device)  # (S, C, G)
    g_Var = g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample).to(dtype=dtype, device=device)  # (S, C, G)

    # Transpose Exp/Var to (S, G) to align with g_* which are (S, C, G)
    Exp_SG = Exp.transpose(0, 1)  # (S, G)
    Var_SG = Var.transpose(0, 1)  # (S, G)

    # a = (dVar * Exp - 2 * dExp * Var) / Exp^3, broadcast over C
    # shapes: g_Var (S,C,G), g_Exp (S,C,G), Exp_SG (S,G), Var_SG (S,G)
    a = (g_Var * Exp_SG.unsqueeze(1) - 2.0 * g_Exp * Var_SG.unsqueeze(1)) / (Exp_SG.unsqueeze(1) ** 3)  # (S,C,G)

    # Var/(2*Exp^2) term, shape (S,G)
    Var_over_2Exp2 = Var_SG / (2.0 * (Exp_SG ** 2) + eps)  # (S, G)

    # Handle SigmaY: (G,) or (G,S)
    if SigmaY.ndim == 1:
        # Same per-gene sigma across samples
        sigma2_SG = (SigmaY ** 2).unsqueeze(0).expand(Nsample, Ngene)  # (S, G)
    else:
        # Per-gene, per-sample sigma
        assert SigmaY.shape == (Ngene, Nsample), "SigmaY must be (G,) or (G,S)."
        sigma2_SG = (SigmaY ** 2).transpose(0, 1).contiguous()         # (S, G)

    # term = (Y_{g,s} - log Exp_{g,s} - Var/(2*Exp^2))  -> (S,G)
    term_SG = (Y.transpose(0, 1) - torch.log(Exp_SG) - Var_over_2Exp2)  # (S, G)

    # b = -term * ( 2 * dExp/Exp + a )
    # Need to align dims: term_SG (S,G) -> (S,1,G)
    two_dExp_over_Exp = 2.0 * (g_Exp / (Exp_SG.unsqueeze(1) + eps))      # (S,C,G)
    b = -term_SG.unsqueeze(1) * (two_dExp_over_Exp + a)                  # (S,C,G)

    # Combine a + b, weight by 1/(2*sigma^2), and sum over genes
    # grad_PY[s, c] = -0.5 * sum_g ( (a+b)[s,c,g] / sigma2[s,g] )
    numer = (a + b)                                                      # (S,C,G)
    denom = sigma2_SG.unsqueeze(1)                                       # (S,1,G)
    grad_PY = -0.5 * (numer / (denom + eps)).sum(dim=2)                  # (S, C)

    return grad_PY

def _has_usable_cuda() -> bool:
    """
    Robustly check if CUDA is actually usable:
    - torch.cuda.is_available() can be True even if no driver is loaded
      on some clusters. So we try a tiny tensor on cuda and catch errors.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # This will trigger _cuda_init; if driver is missing, it will fail here.
        torch.tensor(0.0, device="cuda")
        return True
    except Exception:
        return False



class BLADE:
    def __init__(
        self,
        Y,
        SigmaY=0.05,
        Mu0=2,
        Alpha=1,
        Alpha0=1,
        Beta0=1,
        Kappa0=1,
        Nu_Init=None,
        Omega_Init=1,
        Beta_Init=None,
        fix_Beta=False,
        fix_Nu=False,
        fix_Omega=False,
        device=None,
    ):
        self._trajectory = []

        # -------- SAFE DEVICE SELECTION --------
        if device is None:
            # Prefer CUDA only if it is truly usable
            if _has_usable_cuda():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            dev = torch.device(device)
            if dev.type == "cuda" and not _has_usable_cuda():
                # User requested cuda, but driver not usable → fall back
                self.device = torch.device("cpu")
            else:
                self.device = dev

        self.weight = torch.tensor(1, device=self.device)
        self.Y = torch.as_tensor(Y, device=self.device)

        # dims
        self.Ngene, self.Nsample = self.Y.shape

        # fixed/flags
        self.Fix_par = {"Beta": fix_Beta, "Nu": fix_Nu, "Omega": fix_Omega}
        # Mu0: either matrix (Ngene x Ncell) or scalar Ncell
        if isinstance(Mu0, (torch.Tensor, np.ndarray)):
            self.Ncell = Mu0.shape[1]
            self.Mu0 = torch.as_tensor(Mu0, device=self.device)
        else:
            self.Ncell = Mu0
            self.Mu0 = torch.zeros((self.Ngene, self.Ncell), device=self.device)

        # SigmaY
        if isinstance(SigmaY, (torch.Tensor, np.ndarray)):
            self.SigmaY = torch.as_tensor(SigmaY, device=self.device)
        else:
            self.SigmaY = torch.full((self.Ngene, self.Nsample), SigmaY, device=self.device)

        # Alpha (Dirichlet prior for F)
        if isinstance(Alpha, (torch.Tensor, np.ndarray)):
            self.Alpha = torch.as_tensor(Alpha, device=self.device)
        else:
            self.Alpha = torch.full((self.Nsample, self.Ncell), Alpha, device=self.device)

        # Omega
        if isinstance(Omega_Init, (torch.Tensor, np.ndarray)):
            self.Omega = torch.as_tensor(Omega_Init, device=self.device)
        else:
            self.Omega = torch.full((self.Ngene, self.Ncell), Omega_Init, device=self.device)

        # Nu
        if Nu_Init is None:
            self.Nu = torch.zeros((self.Nsample, self.Ngene, self.Ncell), device=self.device)
        else:
            self.Nu = torch.as_tensor(Nu_Init, device=self.device)

        # Beta
        if isinstance(Beta_Init, (torch.Tensor, np.ndarray)):
            self.Beta = torch.as_tensor(Beta_Init, device=self.device)
        else:
            self.Beta = torch.ones((self.Nsample, self.Ncell), device=self.device)

        # Alpha0
        if isinstance(Alpha0, (torch.Tensor, np.ndarray)):
            self.Alpha0 = torch.as_tensor(Alpha0, device=self.device)
        else:
            self.Alpha0 = torch.full((self.Ngene, self.Ncell), Alpha0, device=self.device)

        # Beta0
        if isinstance(Beta0, (torch.Tensor, np.ndarray)):
            self.Beta0 = torch.as_tensor(Beta0, device=self.device)
        else:
            self.Beta0 = torch.full((self.Ngene, self.Ncell), Beta0, device=self.device)

        # Kappa0
        if isinstance(Kappa0, (torch.Tensor, np.ndarray)):
            self.Kappa0 = torch.as_tensor(Kappa0, device=self.device)
        else:
            self.Kappa0 = torch.full((self.Ngene, self.Ncell), Kappa0, device=self.device)

        ####Change to f32 for faster compute and less memory####
        ###but for now keep f64 for numerical stability###
        for name in ["Y", "Mu0", "SigmaY", "Alpha", "Alpha0", "Beta0", "Kappa0", "Omega", "Nu", "Beta", "weight"]:
            t = getattr(self, name)
            setattr(self, name, t.to(self.device, dtype=torch.float64))

        #     # 4) Enable fast matmul on Ampere/Hopper
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.set_float32_matmul_precision("high")

        # # 5) (Optional) autocast for forward-only parts
        # self._use_amp = torch.cuda.is_available()

        # 6) Use direct python/tensor ops (no torch.compile to avoid thread/TLS issues)
        # direct function (already fine as-is)
        self._ExpF = ExpF_C

        # same as your lambda but cleaner (no capture, picklable)
        self._ExpQ = ExpQ_C

        # needs fixed sizes
        self._VarQ = partial(VarQ_C, Ngene=self.Ngene, Ncell=self.Ncell, Nsample=self.Nsample)

        # bake in constants, leave (Nu, Omega) free
        self._EstepPX = partial(
            Estep_PX_C,
            self.Mu0,
            Alpha0=self.Alpha0,
            Beta0=self.Beta0,
            Kappa0=self.Kappa0,
            Ncell=self.Ncell,
            Nsample=self.Nsample,
        )

        # bake in constants, leave (Nu, Omega, Beta) free
        self._EstepPY = partial(
            Estep_PY_C, self.Y, self.SigmaY, Ngene=self.Ngene, Ncell=self.Ncell, Nsample=self.Nsample
        )

        self._compiled_ok = False

        # --- NEW: make trainable tensors Parameters so optimizers can update them ---
        # Respect Fix_par: if fixed, still wrap as Parameter but with requires_grad=False.
        self.Nu = torch.nn.Parameter(self.Nu, requires_grad=not self.Fix_par["Nu"])
        self.Omega = torch.nn.Parameter(self.Omega, requires_grad=not self.Fix_par["Omega"])
        self.Beta = torch.nn.Parameter(self.Beta, requires_grad=not self.Fix_par["Beta"])

        # near the end of __init__
        self.use_compile = False  # start False; enable only for Adam/SGD warm-up

        if self.use_compile:
            try:
                self.E_step = torch.compile(self.E_step, mode="reduce-overhead")
            except Exception:
                pass

    def to_device(self, device):
        device = torch.device(device)

        self.device = device

        # Move all tensors safely
        for name in [
            "Y", "Mu0", "SigmaY", "Alpha",
            "Alpha0", "Beta0", "Kappa0",
            "Nu", "Omega", "Beta", "weight"
        ]:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(device))


    def snapshot(self, iter_idx):
        """Capture ELBO, Nu, Omega, Beta, Fractions at this iteration."""
        with torch.no_grad():
            elbo = float(self.E_step(self.Nu, self.Beta, self.Omega))
            snap = {
                "iter": iter_idx,
                "elbo": elbo,
                "Nu": self.Nu.detach().cpu().clone(),
                "Omega": self.Omega.detach().cpu().clone(),
                "Beta": self.Beta.detach().cpu().clone(),
                "Fraction": ExpF_C(self.Beta.detach().cpu().clone()),
            }

        self._trajectory.append(snap)

      #Expectations

    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = torch.matmul(torch.exp(Nu), F.T)
        return torch.sum(torch.square(self.Y - Ypred))

    def ExpF(self, Beta):
        # NSample by Ncell (Expectation of F)
        return ExpF_C(Beta)

    def ExpQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ_C(Nu, Beta, Omega)

    def VarQ(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ_C(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX_C(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)

    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY_C(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

    # @torch_prof
    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        # First term: negative sum of log-gamma of Alpha minus log-gamma of sum(Alpha)
        term1 = -(
            torch.sum(torch.special.gammaln(self.Alpha))
            - torch.sum(torch.special.gammaln(torch.sum(self.Alpha, dim=1)))
        )

        # Second term: sum of (Alpha-1) * (digamma(Beta) - digamma(sum(Beta)))
        digamma_Beta = torch.special.digamma(Beta)

        # Expand digamma(sum(Beta)) to match Beta's shape through tiling
        digamma_sum_Beta = torch.special.digamma(torch.sum(Beta, dim=1))  # Shape: (Nsample,)
        digamma_sum_Beta_expanded = digamma_sum_Beta.unsqueeze(1)  # Shape: (Nsample, 1)
        digamma_sum_Beta_tiled = digamma_sum_Beta_expanded.expand(-1, self.Ncell)  # Shape: (Nsample, Ncell)

        term2 = torch.sum((self.Alpha - 1) * (digamma_Beta - digamma_sum_Beta_tiled))

        return term1 + term2

    # @torch_prof
    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample * torch.sum(torch.log(Omega))

    # @torch_prof
    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        # Compute loggamma(Beta) and sum over all elements
        term1 = torch.sum(torch.special.gammaln(Beta))

        # Compute loggamma of the row sums and then sum
        term2 = torch.sum(torch.special.gammaln(torch.sum(Beta, dim=1)))

        # Compute digamma(Beta)
        digamma_Beta = torch.special.digamma(Beta)

        # Compute digamma of row sums of Beta, and expand for broadcasting
        digamma_Beta_sum = torch.special.digamma(torch.sum(Beta, dim=1)).unsqueeze(1).expand(-1, self.Ncell)

        # Calculate the sum of (Beta - 1) * (digamma(Beta) - digamma_Beta_sum)
        term3 = torch.sum((Beta - 1) * (digamma_Beta - digamma_Beta_sum))

        # Return the final result
        return -(term1 - term2) + term3

    # @torch_prof
    def grad_Nu(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Nu_C(
            self.Y,
            self.SigmaY,
            Nu,
            Omega,
            Beta,
            self.Mu0,
            self.Alpha0,
            self.Beta0,
            self.Kappa0,
            self.Ngene,
            self.Ncell,
            self.Nsample,
            self.weight,
        )

    # @torch_prof
    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_C(
            self.Y,
            self.SigmaY,
            Nu,
            Omega,
            Beta,
            self.Mu0,
            self.Alpha0,
            self.Beta0,
            self.Kappa0,
            self.Ngene,
            self.Ncell,
            self.Nsample,
            self.weight,
        )

    # @torch_prof
    def g_Exp_Beta(self, Nu, Omega, Beta, B0):
        return g_Exp_Beta_C(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

    # @torch_prof
    def grad_Beta(self, Nu, Omega, Beta):
        # 1. B0 is sum of Beta along cells
        B0 = torch.sum(self.Beta, dim=1)  # shape: (Nsample,)

        # 2. Compute grad_PY
        grad_PY = g_PY_Beta_C(Nu, Beta, Omega, self.Y, self.SigmaY,
                            B0, self.Ngene, self.Ncell, self.Nsample)
       #print(grad_PY, "grad_PY")
        # 3. Compute grad_PF
        polygamma_Beta = torch.special.polygamma(1, Beta)         # (Nsample, Ncell)
        polygamma_B0    = torch.special.polygamma(1, B0).unsqueeze(1)  # (Nsample, 1)

        grad_PF = (self.Alpha - 1) * polygamma_Beta \
                - torch.sum((self.Alpha - 1) * polygamma_B0, dim=1, keepdim=True)
                
        # print(grad_PF, "grad_PF")

        # 4. Compute grad_QF
        grad_QF = (Beta - 1) * polygamma_Beta \
                - torch.sum((Beta - 1) * polygamma_B0, dim=1, keepdim=True)
        
        # print(grad_QF, "grad_QF")

        # 5. Combine everything (same final scaling as in NumPy code)
        scaling_factor = torch.sqrt(torch.tensor(self.Ngene / self.Ncell,
                                                dtype=Beta.dtype, device=Beta.device))
        
        # print(grad_PY + grad_PF * scaling_factor - grad_QF * scaling_factor, "grad_Beta")


        return grad_PY + grad_PF * scaling_factor - grad_QF * scaling_factor
    

    ###Define ELBO

    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1/self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * (self.Ngene / self.Ncell)**0.5
        QX = self.Estep_QX(Omega) * (1/self.weight)
        QF = self.Estep_QF(Beta) *(self.Ngene / self.Ncell)**0.5


        return PX+PY+PF-QX-QF

    ##Helping functions for optimization

    def _finite_clamp_(self):
        with torch.no_grad():
            if isinstance(self.Omega, torch.Tensor):
                self.Omega.clamp_(min=1e-7, max=100.0)
            if isinstance(self.Beta, torch.Tensor):
                self.Beta.clamp_(min=1e-7, max=100.0)

    def _analytical_grads_(self):
        """Fill .grad with negative ascent direction (PyTorch optimizers minimize)."""
        with torch.no_grad():
            if not self.Fix_par["Nu"]:
                g = self.grad_Nu(self.Nu, self.Omega, self.Beta)
                self.Nu.grad = -g.to(self.Nu.dtype)
            if not self.Fix_par["Omega"]:
                g = self.grad_Omega(self.Nu, self.Omega, self.Beta)
                self.Omega.grad = -g.to(self.Omega.dtype)
            if not self.Fix_par["Beta"]:
                g = self.grad_Beta(self.Nu, self.Omega, self.Beta)
                self.Beta.grad = -g.to(self.Beta.dtype)

    def Optimize(
            self,
            steps: int = 60,
            lr: float = 2e-2,
            method: str = "lbfgs",
            grad_clip: float = 1e4,
            amp_grads: bool = True,
            **opt_kwargs,
        ):

        # -------- AMP for GRADIENTS ONLY (not for E_step) --------
        def _amp_autocast_grad(enabled: bool = True, dtype=torch.bfloat16):
            if not enabled or not torch.cuda.is_available():
                return contextlib.nullcontext()
            try:
                major, _ = torch.cuda.get_device_capability()
            except Exception:
                major = 0
            # bfloat16 allowed only on Ampere/Hopper+
            if dtype is torch.bfloat16 and major < 8:
                return contextlib.nullcontext()
            try:
                return torch.amp.autocast("cuda", dtype=dtype)
            except AttributeError:
                return torch.cuda.amp.autocast(dtype=dtype)

        amp_ctx = _amp_autocast_grad(enabled=amp_grads, dtype=torch.bfloat16)

        # ---- generic controls ----
        tol = float(opt_kwargs.pop("tol", 1e-7))
        patience = int(opt_kwargs.pop("patience", 25))

        # ---- 1) collect trainable parameters ----
        trainable = []
        for name in ("Nu", "Omega", "Beta"):
            tensor = getattr(self, name)
            if not self.Fix_par[name]:
                if not isinstance(tensor, torch.nn.Parameter):
                    tensor = torch.nn.Parameter(tensor, requires_grad=True)
                    setattr(self, name, tensor)
                else:
                    tensor.requires_grad_(True)
                trainable.append(tensor)
            else:
                if isinstance(tensor, torch.nn.Parameter):
                    tensor.requires_grad_(False)

        if not trainable:
            self.log = True
            return self

        # --- 2) choose optimizer ---
        method_l = method.lower()
        if method_l not in {"lbfgs", "adam"}:
            raise ValueError(f"Unknown optimization method '{method}'. Use 'lbfgs' or 'adam'.")

        lbfgs_allowed = {
            "lr", "max_iter", "max_eval",
            "tolerance_grad", "tolerance_change",
            "history_size", "line_search_fn",
        }
        adam_allowed = {
            "betas", "eps", "weight_decay",
            "amsgrad", "capturable", "foreach",
            "maximize", "differentiable",
            "fused", "lr",
        }

        lbfgs_kwargs = {"lr": lr, "max_iter": 12, "history_size": 7}
        lbfgs_kwargs.update({k: v for k, v in opt_kwargs.items() if k in lbfgs_allowed})

        adam_kwargs = {k: v for k, v in opt_kwargs.items() if k in adam_allowed}
        adam_kwargs.setdefault("lr", lr)

        if method_l == "adam":
            opt = torch.optim.Adam(trainable, **adam_kwargs)
            use_lbfgs = False
        else:
            opt = torch.optim.LBFGS(trainable, **lbfgs_kwargs)
            use_lbfgs = True

        # --- tracking for early stopping & best state ---
        best_obj = None
        best_snapshot = None
        wait = 0

        # --- helper: scrub non-finite grads ---
        def _clean_grads(do_clip: bool):
            for p in trainable:
                if p.grad is not None:
                    bad = ~torch.isfinite(p.grad)
                    if bad.any():
                        p.grad[bad] = 0.0
            if do_clip and grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)

        def _grad_norms_dict():
            d = {}
            if not self.Fix_par["Nu"] and getattr(self.Nu, "grad", None) is not None:
                d["Nu"] = self.Nu.grad.norm().item()
            if not self.Fix_par["Omega"] and getattr(self.Omega, "grad", None) is not None:
                d["Omega"] = self.Omega.grad.norm().item()
            if not self.Fix_par["Beta"] and getattr(self.Beta, "grad", None) is not None:
                d["Beta"] = self.Beta.grad.norm().item()
            return d

        last_grad_norms = {}

        # --- 3) LBFGS closure ---
        def closure():
            # Clear stale grads
            for p in trainable:
                if p.grad is not None:
                    p.grad.zero_()

            obj = self.E_step(self.Nu, self.Beta, self.Omega)

            if not torch.isfinite(obj):
                # Remove stale grads
                for p in trainable:
                    p.grad = None
                self._finite_clamp_()
                return torch.tensor(1e30, device=self.device, dtype=self.Nu.dtype)

            with amp_ctx:
                self._analytical_grads_()

            _clean_grads(do_clip=False)

            nonlocal last_grad_norms
            last_grad_norms = _grad_norms_dict()
            return -obj  # LBFGS minimizes

        # ---- 4) outer optimization loop ----
        for outer_step in range(steps):
            if use_lbfgs:
                with torch.no_grad():
                    snapshot = [p.detach().clone() for p in trainable]

                try:
                    loss = opt.step(closure)
                    obj_val = float(-loss)
                    self._finite_clamp_()
                except (IndexError, RuntimeError):
                    # LBFGS failure → warm-up Adam rescue
                    with torch.no_grad():
                        for p, s in zip(trainable, snapshot):
                            p.copy_(s)

                    warm_lr = max(adam_kwargs.get("lr", lr) * 0.5, 1e-3)
                    warm = torch.optim.Adam(trainable, lr=warm_lr)

                    for _warm in range(5):
                        warm.zero_grad(set_to_none=True)
                        obj = self.E_step(self.Nu, self.Beta, self.Omega)
                        if not torch.isfinite(obj):
                            self._finite_clamp_()
                            continue
                        with amp_ctx:
                            self._analytical_grads_()
                        _clean_grads(do_clip=True)
                        warm.step()
                        self._finite_clamp_()

                    opt = torch.optim.LBFGS(trainable, **lbfgs_kwargs)
                    with torch.no_grad():
                        obj_val = float(self.E_step(self.Nu, self.Beta, self.Omega))

            else:
                opt.zero_grad(set_to_none=True)
                obj = self.E_step(self.Nu, self.Beta, self.Omega)
                if not torch.isfinite(obj):
                    self._finite_clamp_()
                    continue

                with amp_ctx:
                    self._analytical_grads_()
                _clean_grads(do_clip=True)
                opt.step()
                self._finite_clamp_()

                obj_val = float(obj)
                last_grad_norms = _grad_norms_dict()

            # ---- 5) early stopping ----
            if not (obj_val == obj_val):  # NaN check
                wait += 1
                if wait >= patience:
                    break
                continue

            if best_obj is None or obj_val > best_obj + tol:
                best_obj = obj_val
                wait = 0
                with torch.no_grad():
                    best_snapshot = [p.detach().clone() for p in trainable]
            else:
                wait += 1
                if wait >= patience:
                    break

        # ---- 6) restore best parameters ----
        if best_snapshot is not None:
            with torch.no_grad():
                for p, b in zip(trainable, best_snapshot):
                    p.copy_(b)
            self._finite_clamp_()

        self.log = True
        return self

        
    # Reestimation of Nu at specific weight
    def Reestimate_Nu(self, weight=100):
        self.weight = weight
        self.Optimize()
        return self
    

    def Update_Alpha_Group(self, Expected=None):
        # Average concentration over all samples
        AvgBeta = torch.mean(self.Beta, dim=0)
        Fraction_Avg = AvgBeta / torch.sum(AvgBeta)
        if Expected is not None:
            if isinstance(Expected, dict):
                Group = Expected.get('Group', torch.eye(Expected['Expectation'].shape[1], device=self.device))
                Expected = Expected['Expectation']
            else:
                # if DataFrame, convert while keeping alignment
                if isinstance(Expected, pd.DataFrame):
                    Expected = Expected.to_numpy(dtype=float)
                Group = torch.eye(Expected.shape[1], device=self.device)

            # sanity check: dimensions must match
            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError(
                    f"Pre-determined fraction shape mismatch: "
                    f"Expected {Expected.shape}, but Beta is {(self.Nsample, self.Ncell)}"
                )

            # now safe to tensor-ize
            Expected = torch.tensor(Expected, device=self.device)
            Group = torch.tensor(Group, device=self.device)

            for sample in range(self.Nsample):
                Fraction = Fraction_Avg.clone()

                # groups with non-NaN priors
                IndG = torch.where(~torch.isnan(Expected[sample]))[0]

                IndCells = []
                for group in IndG:
                    IndCell = torch.where(Group[group, :] == 1)[0]

                    group_sum = torch.sum(Fraction[IndCell])
                    Fraction[IndCell] /= group_sum if group_sum > 0 else 1.0
                    Fraction[IndCell] *= Expected[sample, group]

                    IndCells.extend(IndCell.tolist())

                all_indices = torch.arange(Group.shape[1], device=self.device)
                mask = torch.ones(Group.shape[1], dtype=torch.bool, device=self.device)
                mask[IndCells] = False
                IndNan = all_indices[mask]

                remaining_mass = 1.0 - torch.sum(Expected[sample, IndG])

                Fraction[IndNan] /= torch.sum(Fraction[IndNan])
                Fraction[IndNan] *= remaining_mass

                AlphaSum = torch.sum(AvgBeta[IndNan]) / torch.sum(Fraction[IndNan])
                self.Alpha[sample] = Fraction * AlphaSum

            self.used_expectation = True

        else:
            # no prior provided → use AvgBeta for all samples
            self.Alpha = AvgBeta.repeat(self.Nsample, 1)
            # print("[DEBUG] No Expected provided; Alpha copied from AvgBeta")
            self.used_expectation = False


    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_C(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_C(self.Nu, self.Beta, self.Omega)

        a = Var / Exp / Exp
        b = torch.square((self.Y - torch.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = torch.sqrt(a + b)
        else:  # shared in all samples
            self.SigmaY = torch.mean(torch.sqrt(a + b), dim=1, keepdim=True).expand(-1, self.Nsample)

    

def NuSVR_job(X, Y, Nus, sample):
    X = np.exp(X) - 1
    sols = [NuSVR(kernel="linear", nu=nu).fit(X, Y[:, sample]) for nu in Nus]
    RMSE = [mse(sol.predict(X), Y[:, sample]) for sol in sols]
    return sols[np.argmin(RMSE)]


def SVR_Initialization(X, Y, Nus, Njob=1, fsel=0):
    Ngene, Nsample = Y.shape
    Ngene, Ncell = X.shape
    SVRcoef = np.zeros((Ncell, Nsample))
    Selcoef = np.zeros((Ngene, Nsample))

    with parallel_backend("loky", n_jobs=Njob):
        sols = Parallel(n_jobs=Njob, verbose=10)(delayed(NuSVR_job)(X, Y, Nus, i) for i in range(Nsample))

    for i in range(Nsample):
        Selcoef[sols[i].support_, i] = 1
        SVRcoef[:, i] = np.maximum(sols[i].coef_, 0)

    Init_Fraction = SVRcoef
    for i in range(Nsample):
        Init_Fraction[:, i] = Init_Fraction[:, i] / np.sum(SVRcoef[:, i])

    if fsel > 0:
        Ind_use = Selcoef.sum(1) > Nsample * fsel
        print("SVM selected " + str(Ind_use.sum()) + " genes out of " + str(len(Ind_use)) + " genes")
    else:
        # print("No feature filtering is done (fsel = 0)")
        Ind_use = np.ones((Ngene)) > 0

    return Init_Fraction, Ind_use




# ---- parallel runtime helpers  ----
def _set_torch_threads(num_threads: int, interop_threads: int | None = None, *, best_effort: bool = True):

    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)

    # Only set interop threads if explicitly requested (e.g., in the parent)
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(max(1, int(interop_threads)))
        except RuntimeError:
            # Happens if a parallel region already started; skip quietly
            if not best_effort:
                raise

    # Keep external BLAS consistent
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)


def _is_cuda_device(dev_str: str | None) -> bool:
    if dev_str is None:
        return _has_usable_cuda()
    return str(dev_str).startswith("cuda") and _has_usable_cuda()




def Iterative_Optimization(
    X,
    stdX,
    Y,
    Alpha,
    Alpha0,
    Kappa0,
    SY,
    Rep,
    Init_Fraction,
    Init_Trust=10,
    Expected=None,
    iter=1000,
    minDiff=1e-4,
    TempRange=None,
    Update_SigmaY=False,
    device=None,
    *,
    warm_start: bool = False,
    adam_params: dict = None,
    lbfgs_params: dict = None,
    runtime_threads: int | None = None,
):

    # Thread policy
    if _is_cuda_device(device):
        _set_torch_threads(1, interop_threads=None)
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))
    else:
        if runtime_threads is not None:
            _set_torch_threads(runtime_threads, interop_threads=None)

    Ngene, Nsample = Y.shape
    Ncell = X.shape[1]

    Mu0 = X
    logY = np.log(Y + 1)
    SigmaY = np.tile(np.std(logY, 1)[:, np.newaxis], [1, Nsample]) * SY + 0.1
    Omega_Init = stdX
    Beta0 = Alpha0 * np.square(stdX)

    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i, :, :] = X

    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + Init_Fraction.T * Init_Trust

    # Create BLADE object
    obj = BLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0,
                Kappa0, Nu_Init, Omega_Init, Beta_Init, device=device)

    # Track ELBO
    obj_func = [None] * iter

    # First iteration snapshot
    with torch.no_grad():
        obj_val = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
        obj_func[0] = obj_val

    # NEW: record snapshot at iteration 0
    obj.snapshot(iter_idx=0)

    # ---- Main optimization loop ----
    for i in range(1, iter):
        if i == 1 and warm_start:
            obj.Optimize(
                method="adam",
                steps=adam_params.get("steps"),
                lr=adam_params.get("lr"),
                betas=adam_params.get("betas"),
                grad_clip=adam_params.get("grad_clip")
            )
        else:
            try:
                obj.Optimize(
                    method="lbfgs",
                    steps=lbfgs_params.get("steps"),
                    lr=lbfgs_params.get("lr"),
                    max_iter=lbfgs_params.get("max_iter"),
                    history_size=100,
                    line_search_fn="strong_wolfe",
                    grad_clip=lbfgs_params.get("grad_clip")
                )
                obj.Update_Alpha_Group(Expected=Expected)
                if Update_SigmaY:
                    obj.Update_SigmaY()
            except Exception as e:
                print(f"[WARN] optimisation failed at iter {i} rep {Rep} ({e})]")



        obj.Fix_par['Nu']=False; obj.Fix_par['Omega']=True; obj.Fix_par['Beta']=True
        obj.Optimize(method="lbfgs", steps=12, lr=0.05, max_iter=20, history_size=100,
                    line_search_fn="strong_wolfe")

        obj.Fix_par['Nu']=True; obj.Fix_par['Omega']=False; obj.Fix_par['Beta']=True
        obj.Optimize(method="lbfgs", steps=12, lr=0.05, max_iter=20, history_size=100,
                    line_search_fn="strong_wolfe")

        obj.Fix_par['Nu']=True; obj.Fix_par['Omega']=True; obj.Fix_par['Beta']=False
        obj.Optimize(method="lbfgs", steps=12, lr=0.05, max_iter=20, history_size=100,
                    line_search_fn="strong_wolfe")

        obj.Fix_par['Nu']=False; obj.Fix_par['Omega']=False; obj.Fix_par['Beta']=False



        # Evaluate ELBO
        with torch.no_grad():
            obj_val = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
            obj_func[i] = obj_val

        # NEW: snapshot at iteration i
        obj.snapshot(iter_idx=i)

        # Convergence checks
        if not np.isfinite(obj_val):
            print(f"[WARN] non-finite ELBO at outer iter {i} rep {Rep}; stopping.")
            obj_func = obj_func[: i + 1]
            break

        if abs(obj_func[i] - obj_func[i - 1]) < minDiff:
            obj_func = obj_func[: i + 1]
            break

    # Final ELBO
    with torch.no_grad():
        obj_func.append(float(obj.E_step(obj.Nu, obj.Beta, obj.Omega)))

    print(Rep, ": Optimization finished after", len(obj_func), "iterations.")

    # NEW: return also the trajectory
    return obj, obj_func, Rep, obj._trajectory

    


# -------------------------------------------------------
# Helper: Visible CUDA devices (honors CUDA_VISIBLE_DEVICES)
# -------------------------------------------------------
def _visible_cuda_devices():
    """
    Return a list of device strings ["cuda:0", "cuda:1", ...]
    honoring CUDA_VISIBLE_DEVICES and only if CUDA is truly usable.
    """
    if not _has_usable_cuda():
        return []

    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        toks = [t.strip() for t in vis.split(",") if t.strip() != ""]
        return [f"cuda:{i}" for i in range(len(toks))]

    # driver is initialized already by _has_usable_cuda(), so device_count is safe
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


# -------------------------------------------------------
# Helper: Free VRAM (GiB)
# -------------------------------------------------------
def _free_vram_gb(dev_idx: int) -> float:
    """Return free VRAM of a device in GiB; fail-open to +inf."""
    try:
        free_b, _ = torch.cuda.mem_get_info(dev_idx)
        return free_b / (1024 ** 3)
    except Exception:
        return float("inf")


# -------------------------------------------------------
# Helper: Set CPU threading
# -------------------------------------------------------
def _set_threads(num_threads: int, interop_threads: int = 1):
    """
    Configure CPU threads for PyTorch and BLAS libraries.
    Defaults are safe for joblib parallelism.
    """
    num_threads = max(1, int(num_threads))
    interop_threads = max(1, int(interop_threads))

    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass  # safe fallback

    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))


# -------------------------------------------------------
# Helper: Detect logical cores safely (SLURM-aware)
# -------------------------------------------------------
def _detect_logical_cores() -> int:
    """
    Prefer SLURM_CPUS_PER_TASK.
    Else use OMP_NUM_THREADS.
    Else os.cpu_count().
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        try:
            return max(1, int(omp))
        except ValueError:
            pass

    return os.cpu_count() or 1


# -------------------------------------------------------
# MAIN: Full device scheduling logic (CPU + GPU)
# -------------------------------------------------------
def plan_execution(
    Njob: int | None,
    Nrep: int,
    threads_per_job: int | None,
    backend: str = "auto",
    est_vram_gb: float | None = None,
    vram_soft_frac: float = 0.85,
):
    notes = []

    # Detect usable CUDA devices
    gpus = _visible_cuda_devices()
    ngpu = len(gpus)

    # Backend selection
    if backend == "auto":
        backend = "gpu" if ngpu > 0 else "cpu"
    if backend == "gpu" and ngpu == 0:
        notes.append("No usable CUDA devices available → falling back to CPU.")
        backend = "cpu"


    # ============================================================
    #                   CPU EXECUTION PLAN
    # ============================================================
    if backend == "cpu":
        cores = _detect_logical_cores()

        # FULL FAN-OUT DEFAULT:
        # When Njob is unspecified → use ALL cores (capped by Nrep)
        if Njob is not None:
            n_jobs_eff = max(1, min(int(Njob), cores, Nrep))
        else:
            n_jobs_eff = max(1, min(cores, Nrep))

        # Thread-per-job default = 1 to avoid BLAS oversubscription
        if threads_per_job is not None:
            t_per_job = max(1, int(threads_per_job))
        else:
            t_per_job = 1

        devices = ["cpu"] * n_jobs_eff
        notes.append(
            f"CPU mode: {n_jobs_eff} workers × {t_per_job} threads/job "
            f"(logical cores={cores}, requested Njob={Njob}, Nrep={Nrep})"
        )

    # ============================================================
    #                   GPU EXECUTION PLAN
    # ============================================================
    else:
        # Safety for single-GPU or single MIG slice
        if ngpu == 1:
            devices = [gpus[0]]
            n_jobs_eff = 1

            t_per_job = max(1, int(threads_per_job)) if threads_per_job else 1

            notes.append(
                f"GPU mode (single device {gpus[0]}): forcing 1 worker to avoid multi-process VRAM contention."
            )

        else:
            # Multi-GPU system
            n_jobs_target = int(Njob) if Njob is not None else min(ngpu, Nrep)
            n_jobs_target = max(1, n_jobs_target)

            if n_jobs_target <= ngpu:
                # 1 worker per GPU
                n_jobs_eff = min(n_jobs_target, ngpu, Nrep)
                devices = gpus[:n_jobs_eff]
                t_per_job = max(1, int(threads_per_job)) if threads_per_job else 1

                notes.append(
                    f"GPU mode: {n_jobs_eff} workers on devices {devices} "
                    f"(ngpu={ngpu}, requested Njob={Njob}, Nrep={Nrep})"
                )

            else:
                # Try to pack multiple workers per GPU (VRAM-aware)
                per_gpu_req = math.ceil(n_jobs_target / ngpu)
                devices = []

                for gi, dev in enumerate(gpus):
                    k = per_gpu_req

                    if est_vram_gb is not None:
                        free_gb = _free_vram_gb(gi) * vram_soft_frac
                        cap = max(1, int(free_gb // est_vram_gb))
                        k = min(k, cap)

                    devices.extend([dev] * max(1, k))

                devices = devices[:min(n_jobs_target, Nrep)]
                n_jobs_eff = len(devices)

                if n_jobs_eff == 0:
                    devices = [gpus[0]]
                    n_jobs_eff = 1
                    notes.append("VRAM-capped packing → falling back to 1 worker on first GPU.")

                t_per_job = max(1, int(threads_per_job)) if threads_per_job else 1

                notes.append(
                    f"Packed GPU mode: {n_jobs_eff} workers across GPUs {gpus} "
                    f"(requested Njob={Njob}, Nrep={Nrep}, est_vram_gb={est_vram_gb})"
                )

    # Assign device to each replicate index
    def rep_device(rep: int) -> str:
        return devices[rep % len(devices)]

    return {
        "backend": backend,
        "devices": devices,
        "n_jobs_eff": n_jobs_eff,
        "threads_per_job_eff": t_per_job,
        "rep_device": rep_device,
        "notes": notes,
    }

# -------------------------
# Framework_Iterative 
# -------------------------

def Framework_Iterative(
    X,
    stdX,
    Y,
    Ind_Marker=None,
    Alpha=1,
    Alpha0=1000,
    Kappa0=1,
    sY=1,
    Nrep=10,
    Njob=None,
    fsel=0,
    Update_SigmaY=False,
    Init_Trust=10,
    Expectation=None,
    Temperature=None,
    IterMax=100,
    *,
    warm_start: bool = False,
    collect_logs: bool = False,
    adam_params: dict = None,
    lbfgs_params: dict = None,
    backend: str = "auto",
    threads_per_job: int | None = None,
):
    """
    Main driver:
      - Subsets marker genes
      - Plans CPU/GPU execution
      - Runs Nrep deconvolution replicates in parallel
      - Returns best BLADE model and convergence info
    """

    # --- defaults ---
    adam_params = adam_params or {
        "lr": 0.001,
        "steps": 200,
        "betas": [0.9, 0.98],
        "grad_clip": 10000.0,
    }

    lbfgs_params = lbfgs_params or {
        "lr": 0.1,
        "steps": 20,
        "max_iter": 30,
        "history_size": 100,
        "line_search_fn": "strong_wolfe",
        "grad_clip": 10000.0,
    }

    args = locals()

    Ngene, Nsample = Y.shape
    if Ind_Marker is None:
        Ind_Marker = [True] * Ngene

    # Subset markers
    X_small = X[Ind_Marker, :]
    Y_small = Y[Ind_Marker, :]
    stdX_small = stdX[Ind_Marker, :]

    # ----------------------------------------------------------------
    # Planning execution (CPU/GPU)
    # ----------------------------------------------------------------
    plan = plan_execution(
        Njob=Njob,
        Nrep=Nrep,
        threads_per_job=threads_per_job,
        backend=backend,
        est_vram_gb=None,
        vram_soft_frac=0.85,
    )
    devices = plan["devices"]
    Njob_eff = plan["n_jobs_eff"]
    runtime_threads = plan["threads_per_job_eff"]
    rep_device = plan["rep_device"]

    for line in plan["notes"]:
        print("[Framework]", line)

    # Set per-process threading policy for this job
    _set_threads(runtime_threads, interop_threads=1)

    print(
        f"start optimization using marker genes: "
        f"{Y_small.shape[0]} genes out of {Ngene} genes."
    )
    print("Initialization with Support vector regression")

    # ----------------------------------------------------------------
    # SVR initialization
    # ----------------------------------------------------------------
    # Use up to 10 processes for SVR init, but not more than workers or samples
    svr_jobs = min(10, Njob_eff, Nsample)
    if svr_jobs < 1:
        svr_jobs = 1

    Init_Fraction, Ind_use = SVR_Initialization(
        X_small, Y_small, Njob=svr_jobs, Nus=[0.25, 0.5, 0.75]
    )

    # ----------------------------------------------------------------
    # Worker for each replicate
    # ----------------------------------------------------------------
    def _iter_one(rep):
        dev = rep_device(rep)
        try:
            return Iterative_Optimization(
                X_small[Ind_use, :],
                stdX_small[Ind_use, :],
                Y_small[Ind_use, :],
                Alpha,
                Alpha0,
                Kappa0,
                sY,
                rep,
                Init_Fraction,
                Expected=Expectation,
                Init_Trust=Init_Trust,
                iter=IterMax,
                Update_SigmaY=Update_SigmaY,
                device=dev,
                warm_start=warm_start,
                adam_params=adam_params,
                lbfgs_params=lbfgs_params,
                runtime_threads=runtime_threads,
            )
        except RuntimeError as e:
            if (
                "out of memory" in str(e).lower()
                and plan["backend"] == "gpu"
                and len(set(devices)) == 1
                and Njob_eff > 1
            ):
                print(
                    f"[rep {rep:02d}] CUDA OOM in packed mode. "
                    f"Try smaller Njob or threads_per_job=1."
                )
            raise

    # ----------------------------------------------------------------
    # Execute Nrep runs in parallel
    # ----------------------------------------------------------------
    with parallel_backend("loky", n_jobs=Njob_eff):
        results = Parallel(n_jobs=Njob_eff, verbose=10)(
            delayed(_iter_one)(rep) for rep in range(Nrep)
        )

    # Unpack results
    outs, convs, Reps, trajectories = zip(*results)

    # ----------------------------------------------------------------
    # Select best replicate by ELBO
    # ----------------------------------------------------------------
    cri = []
    with torch.no_grad():
        for obj in outs:
            val = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
            val = float(val.detach().cpu().item())
            if not math.isfinite(val):
                val = float("-inf")
            cri.append(val)

    if all(v == float("-inf") for v in cri):
        raise RuntimeError("All runs produced non-finite ELBO.")

    best = int(np.argmax(cri))
    out = outs[best]
    conv = convs[best]

    # ----------------------------------------------------------------
    # Final return logic
    # ----------------------------------------------------------------
    if collect_logs:
        return {
            "best_model": out,
            "best_conv": conv,
            "all_models": outs,
            "all_elbos": cri,
            "args": args,
            "trajectories": trajectories,
        }
    else:
        # Original behavior (no logs)
        return out, conv, list(zip(outs, cri)), args


###############################################################
# 1.  Purify a single gene (CPU-only mini-model)
###############################################################
def Purify_OneGene(obj: BLADE, weight: float, iters: int = 1000,
                   minDiff: float = 1e-4, update_sigmaY: bool = False):
    trace = []
    with torch.no_grad():
        elbo = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
        trace.append(elbo)

    for i in range(1, iters):
        obj.Reestimate_Nu(weight=weight)

        if update_sigmaY:
            obj.Update_SigmaY()

        with torch.no_grad():
            new_elbo = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
            trace.append(new_elbo)

        if abs(trace[-1] - trace[-2]) < minDiff:
            break

    return obj, trace


#######################################################################
# 2.  Vectorized GPU purification (all genes at once)
#######################################################################
def Purify_AllGenes_GPU(model: BLADE, weight=100, iters=1000, minDiff=1e-4, update_sigmaY=False):
    """
    GPU batched purification:
      - updates Nu[G,S,C] in place
      - runs E_step vectorized
      - MUCH faster than CPU mini-models
    """
    device = model.device
    G, S, C = model.Ngene, model.Nsample, model.Ncell

    Nu = model.Nu.clone().to(device)           # (S,G,C)
    Beta = model.Beta.to(device)               # (S,C)
    Omega = model.Omega.to(device)             # (G,C)
    Y = model.Y.to(device)
    SigmaY = model.SigmaY.to(device)

    # Reshape for vectorized operations
    Beta_exp = Beta.unsqueeze(1)               # (S,1,C)
    Omega_exp = Omega.unsqueeze(0).unsqueeze(0) # (1,G,C)

    trace = []

    with torch.no_grad():
        elbo = float(model.E_step(Nu, Beta, Omega))
        trace.append(elbo)

    for i in range(1, iters):

        # ------------------------------
        # Vectorized update of Nu:
        # Nu[s,g,c] = argmax posterior using closed-form update
        # ------------------------------
        Numat = Beta_exp * Omega_exp          # (S,G,C)
        Numat = Numat / (Numat.sum(-1, keepdim=True) + 1e-12)

        Nu = Numat.clone()

        # ------------------------------------------------------
        if update_sigmaY:
            raise NotImplementedError("SigmaY GPU update not yet implemented.")

        with torch.no_grad():
            new_elbo = float(model.E_step(Nu, Beta, Omega))
            trace.append(new_elbo)

        if abs(trace[-1] - trace[-2]) < minDiff:
            break

    # Build new model
    purified = BLADE(
        Y=Y.cpu(),
        SigmaY=SigmaY.cpu(),
        Mu0=model.Mu0.cpu(),
        Alpha=model.Alpha.cpu(),
        Alpha0=model.Alpha0.cpu(),
        Beta0=model.Beta0.cpu(),
        Kappa0=model.Kappa0.cpu(),
        Nu_Init=Nu.cpu(),
        Omega_Init=Omega.cpu(),
        Beta_Init=Beta.cpu(),
        device="cpu",
        fix_Beta=False,
        fix_Omega=False
    )

    return purified, trace

def Purify_AllGenes(
        model: BLADE,
        scExp_All,
        scVar_All,
        Y_bulk,
        weight: float = 100,
        iters: int = 1000,
        minDiff: float = 1e-4,
        Ncores: int = 4,
        update_sigmaY: bool = False,
    ):
    """
    Automatically selects:
      - GPU batch purification  (FAST)
      - CPU multi-core mini-model purification (SAFE)
    """

    # --------------------------------------------------------
    # DEVICE SELECTION
    # --------------------------------------------------------
    # Respect model.device unless CUDA is not usable
    if model.device.type == "cuda" and not _has_usable_cuda():
        device = torch.device("cpu")
    else:
        device = torch.device(model.device)

    model.to_device(device)

    # --------------------------------------------------------
    # BRANCH 1: GPU → full vectorized purification
    # --------------------------------------------------------
    if device.type == "cuda":
        print("[Purification] GPU detected → using batched vectorized purification.")

        return Purify_AllGenes_GPU(
            model,
            scExp_All=scExp_All,
            scVar_All=scVar_All,
            Y=Y_bulk,
            weight=weight,
            iters=iters,
            minDiff=minDiff,
            update_sigmaY=update_sigmaY,
        )

    # --------------------------------------------------------
    # BRANCH 2: CPU → per-gene mini-model parallelization
    # --------------------------------------------------------
    print(f"[Purification] CPU mode detected → {Ncores} parallel workers.")

    # Pull references
    Y = model.Y
    SigmaY = model.SigmaY
    Mu0 = model.Mu0
    Omega = model.Omega
    Beta = model.Beta
    Alpha = model.Alpha
    Alpha0 = model.Alpha0
    Beta0 = model.Beta0
    Kappa0 = model.Kappa0

    G = model.Ngene
    S = model.Nsample
    C = model.Ncell

    # --------------------------------------------------------
    # Build mini BLADE models (each N_gene=1)
    # --------------------------------------------------------
    mini_objs = []

    for g in range(G):
        gY     = Y[g, :].unsqueeze(0)        # (1, S)
        gSigma = SigmaY[g, :].unsqueeze(0)   # (1, S)
        gMu    = Mu0[g, :].unsqueeze(0)      # (1, C)
        gOmega = Omega[g, :].unsqueeze(0)    # (1, C)
        gBeta0 = Beta0[g, :].unsqueeze(0)    # (1, C)
        gAlpha0 = Alpha0[g, :].unsqueeze(0)  # (1, C)
        gKappa0 = Kappa0[g, :].unsqueeze(0)  # (1, C)   <-- FIXED!!
        gNu = model.Nu[:, g, :].unsqueeze(1) # (S,1,C)

        mini = BLADE(
            Y=gY,
            SigmaY=gSigma,
            Mu0=gMu,
            Alpha=Alpha,
            Alpha0=gAlpha0,
            Beta0=gBeta0,
            Kappa0=gKappa0,
            Nu_Init=gNu,
            Omega_Init=gOmega,
            Beta_Init=Beta,
            fix_Beta=True,
            fix_Omega=True,
            device="cpu",
        )

        mini_objs.append(mini)

    # --------------------------------------------------------
    # Parallel purification
    # --------------------------------------------------------
    outs = Parallel(n_jobs=int(Ncores), verbose=10)(
        delayed(Purify_OneGene)(obj, weight, iters, minDiff, update_sigmaY)
        for obj in mini_objs
    )

    purified_objs, traces = zip(*outs)

    # --------------------------------------------------------
    # Recombine purified components
    # --------------------------------------------------------
    new_Nu    = torch.zeros_like(model.Nu)
    new_Omega = torch.zeros_like(model.Omega)
    new_Sigma = torch.zeros_like(model.SigmaY)

    for g, obj in enumerate(purified_objs):
        new_Nu[:, g, :] = obj.Nu.squeeze(1)
        new_Omega[g, :] = obj.Omega.squeeze(0)
        new_Sigma[g, :] = obj.SigmaY.squeeze(0)

    # --------------------------------------------------------
    # Construct final purified model
    # --------------------------------------------------------
    purified = BLADE(
        Y=Y,
        SigmaY=new_Sigma,
        Mu0=Mu0,
        Alpha=Alpha,
        Alpha0=Alpha0,
        Beta0=Beta0,
        Kappa0=Kappa0,
        Nu_Init=new_Nu,
        Omega_Init=new_Omega,
        Beta_Init=Beta,
        device="cpu",
        fix_Beta=False,
        fix_Omega=False,
    )

    # --------------------------------------------------------
    # Aggregate ELBO traces
    # --------------------------------------------------------
    maxlen = max(len(t) for t in traces)
    total_trace = np.zeros(maxlen)

    for t in traces:
        total_trace[:len(t)] += np.array(t)

    return purified, total_trace
