import os
# Disable CUDA graphs in Inductor — safer under multi-thread/multi-proc joblib
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")


# New imports
import torch
import torch.special

import dill
# Old imports
from numba import jit, njit

import numpy as np
from numpy import transpose as t
import scipy.optimize
from scipy.special import loggamma
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import polygamma

from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

from joblib import Parallel, delayed, parallel_backend
import itertools
import time
import os, math
import warnings

from timeit import default_timer as timer
from functools import partial

import contextlib

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# ---- TorchDynamo guard (version-safe) ----
import importlib
from contextlib import contextmanager

_dynamo_disable = None
try:
    _dynamo_mod = importlib.import_module("torch._dynamo")
    _dynamo_disable = getattr(_dynamo_mod, "disable", None)  # context manager in recent PyTorch
except Exception:
    _dynamo_disable = None

@contextmanager
def maybe_disable_dynamo():
    """No-op if torch._dynamo.disable is unavailable."""
    if _dynamo_disable is None:
        yield
    else:
        with _dynamo_disable():
            yield

# numeric guards
EXP_MAX = 80.0          # cap exponent arguments (float32-safe)
EPS     = 1e-12         # safe minimum for divides/log/exp

# The below function is a decorator to cast numpy arrays to torch tensors
def cast_args_to_torch(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        device = self.device

        new_args = [
            torch.tensor(arg, dtype=torch.float32, device=device)
            if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        new_kwargs = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        return func(*new_args, **new_kwargs)
    return wrapper


# --- CPU fastpath config (no memmaps, production-safe) ---
CPU_CHUNK_G = int(os.environ.get("BLADE_CPU_CHUNK_G", "2048"))

def _use_cpu_fastpath(t: torch.Tensor) -> bool:
    return t.device.type == "cpu"

def _bmm_sum_over_c(weights_sc: torch.Tensor, mat_sgc: torch.Tensor) -> torch.Tensor:
    """
    (S,C) x (S,G,C) -> (S,G) using batched GEMM.
    Much faster than equivalent einsum on CPU.
    """
    return torch.bmm(
        weights_sc.unsqueeze(1),                 # (S,1,C)
        mat_sgc.transpose(1, 2).contiguous()    # (S,C,G)
    ).squeeze(1)                                 # (S,G)





def _safe_exp(x):
    return torch.exp(torch.clamp(x, max=EXP_MAX))



def ExpF_C(Beta):
    # Compute normalized Beta across the second axis (cells).
    return Beta / torch.sum(Beta, dim=1, keepdim=True)


def ExpQ_C(Nu, Beta, Omega):
    # Expected value of F (Nsample by Ncell)
    ExpB = ExpF_C(Beta)

    # Element-wise exponential and broadcasting of Nu and Omega
    out = torch.sum(ExpB.unsqueeze(1) * torch.exp(Nu + 0.5 * Omega**2), dim=-1)
    return out.T

def VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    """
    Variance of Q(Y). Returns (Ngene, Nsample).
    GPU: original vectorized path.
    CPU: tiled + bmm fastpath.
    """
    if _use_cpu_fastpath(Nu):
        return _VarQ_C_cpu(Nu, Beta, Omega, Ngene, Ncell, Nsample, chunk_G=CPU_CHUNK_G)

    # ---- original GPU-friendly code ----
    EPS_ = globals().get("EPS", 1e-6)
    EXP_MAX_ = globals().get("EXP_MAX", 80.0)

    B0  = Beta.sum(dim=1).clamp_min(EPS_)                    # (S,)
    Bt  = Beta / B0.unsqueeze(1)                             # (S,C)
    VarB = Bt * (1.0 - Bt) / (B0 + 1.0).unsqueeze(1)         # (S,C)

    arg2  = torch.clamp(2.0 * Nu + 2.0 * Omega.unsqueeze(0).square(), max=EXP_MAX_)
    exp2  = torch.exp(arg2)                                  # (S,G,C)
    arg2h = torch.clamp(2.0 * Nu +      Omega.unsqueeze(0).square(),   max=EXP_MAX_)
    exp2h = torch.exp(arg2h)                                 # (S,G,C)

    VarTerm = (
        exp2  * (VarB.unsqueeze(1) + Bt.unsqueeze(1).square())
        - exp2h * Bt.unsqueeze(1).square()
    ).sum(dim=-1).T                                          # (G,S)

    argv = torch.clamp(Nu + 0.5 * Omega.unsqueeze(0).square(), max=EXP_MAX_)
    v    = torch.exp(argv)                                   # (S,G,C)

    sum_bt_v   = torch.einsum('sc,sgc->sg', Bt, v)
    sum_bt2_v2 = torch.einsum('sc,sgc->sg', Bt.square(), v.square())
    CovTerm_SG = - (sum_bt_v.square() - sum_bt2_v2) / (1.0 + B0).unsqueeze(1)

    return VarTerm + CovTerm_SG.T


def _VarQ_C_cpu(Nu, Beta, Omega, Ngene, Ncell, Nsample, *, chunk_G: int = 2048):
    """CPU fastpath: tile G and use bmm instead of einsum."""
    EPS_ = globals().get("EPS", 1e-6)
    EXP_MAX_ = globals().get("EXP_MAX", 80.0)
    S, G, C = Nu.shape

    B0  = Beta.sum(dim=1).clamp_min(EPS_)                    # (S,)
    Bt  = Beta / B0.unsqueeze(1)                             # (S,C)
    VarB = Bt * (1.0 - Bt) / (B0 + 1.0).unsqueeze(1)         # (S,C)

    out = torch.empty((G, S), dtype=Nu.dtype, device=Nu.device)

    for g0 in range(0, G, chunk_G):
        g1 = min(G, g0 + chunk_G)
        Nu_s = Nu[:, g0:g1, :]                               # (S,g,C)
        Om_s = Omega[g0:g1, :]                               # (g,C)

        exp2  = torch.exp(torch.clamp(2.0 * Nu_s + 2.0 * Om_s.unsqueeze(0).square(), max=EXP_MAX_))
        exp2h = torch.exp(torch.clamp(2.0 * Nu_s +      Om_s.unsqueeze(0).square(),   max=EXP_MAX_))

        T1 = _bmm_sum_over_c(VarB + Bt.square(), exp2)       # (S,g)
        T2 = _bmm_sum_over_c(Bt.square(),        exp2h)      # (S,g)
        VarTerm_Sg = T1 - T2

        v = torch.exp(torch.clamp(Nu_s + 0.5 * Om_s.unsqueeze(0).square(), max=EXP_MAX_))  # (S,g,C)
        sum_bt_v   = _bmm_sum_over_c(Bt,          v)          # (S,g)
        sum_bt2_v2 = _bmm_sum_over_c(Bt.square(), v.square()) # (S,g)
        Cov_Sg = - (sum_bt_v.square() - sum_bt2_v2) / (1.0 + B0).unsqueeze(1)

        out[g0:g1, :] = (VarTerm_Sg + Cov_Sg).T.contiguous()

    return out




def Estep_PY_C(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_C(Nu, Beta, Omega)

    a = Var / Exp / Exp

    return torch.sum(
            -0.5 / torch.square(SigmaY) * (a + torch.square((Y-torch.log(Exp)) - 0.5 * a))
            )


def Estep_PX_C(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = torch.sum(Nu, 0) / Nsample  # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5 * Nsample  # Posterior Alpha

    ExpBetaN = Beta0 + (Nsample - 1) / 2 * torch.square(Omega) + \
               Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (torch.square(Omega) / Nsample + torch.square(NuExp - Mu0))

    # Vectorize the for loop
    ExpBetaN = ExpBetaN + 0.5 * torch.sum(torch.square(Nu - NuExp.unsqueeze(0)), dim=0)

    return torch.sum(-AlphaN * torch.log(ExpBetaN))


def grad_Nu_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0,
                   Ngene, Ncell, Nsample, weight):
    """Memory-safe grad wrt Nu (no CovX 4-D tensor).  Shapes preserved."""
    # ----- PX term -----
    AlphaN = Alpha0 + 0.5 * Nsample
    NuExp  = torch.sum(Nu, dim=0) / Nsample
    ExpBetaN = Beta0 + (Nsample - 1) * 0.5 * Omega.square() \
             + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (Omega.square()/Nsample + (NuExp - Mu0).square())
    ExpBetaN = ExpBetaN + 0.5 * torch.sum((Nu - NuExp).square(), dim=0)
    DiffMean = torch.mean(Nu - NuExp, dim=0)
    Nominator = Nu - NuExp - DiffMean + Kappa0/(Kappa0 + Nsample) * (NuExp - Mu0)
    grad_PX = -AlphaN * Nominator / ExpBetaN.clamp_min(EPS)

    # ----- PY term (all memory-safe) -----
    B0     = Beta.sum(dim=1).clamp_min(EPS)             # (S,)
    Btilda = Beta / B0.unsqueeze(1)                      # (S,C)

    # Exp, Var (G,S)
    Exp = ExpQ_C(Nu, Beta, Omega).clamp_min(EPS)
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample).clamp_min(0.0)

     # CovB and its diagonal
    CovB  = -torch.einsum('sl,sk->slk', Btilda, Btilda) / (1.0 + B0).unsqueeze(1).unsqueeze(2)  # (S,C,C)
    diagB = CovB.diagonal(dim1=1, dim2=2)                                                       # (S,C)

    # v = E[X] guard
    argx = torch.clamp(Nu + 0.5 * Omega.square().unsqueeze(0), max=EXP_MAX)  # (S,G,C)
    ExpX = torch.exp(argx)                                                    # (S,G,C)

    # g_Exp (G,C,S)
    g_Exp = (ExpX * Btilda.unsqueeze(1)).permute(1, 2, 0)                     # (G,C,S)

    # Var pieces (unchanged) -> g_Var base
    VarX = torch.exp(torch.clamp(2*Nu + 2*Omega.square().unsqueeze(0), max=EXP_MAX))
    VarB = Btilda * (1.0 - Btilda)
    VarB /= (B0 + 1.0).unsqueeze(1)
    first_term  = 2.0 * VarX * (VarB + Btilda.square()).unsqueeze(1)
    diag_CovX   = ExpX.square()
    second_term = 2.0 * diag_CovX * Btilda.square().unsqueeze(1)
    g_Var = first_term.permute(1, 2, 0) - second_term.permute(1, 2, 0)        # (G,C,S)

    # === CovTerm without forming CovX ===
    if _use_cpu_fastpath(Nu):
        v_sgc = ExpX                                       # (S,G,C)
        t_sgc = torch.bmm(v_sgc, CovB)                     # (S,G,C)
        u_sgc = t_sgc - diagB.unsqueeze(1) * v_sgc         # (S,G,C)
        CovTerm = (2.0 * v_sgc * u_sgc).permute(1, 2, 0)   # (G,C,S)
    else:
        v = ExpX.permute(1, 0, 2)                          # (G,S,C)
        t = torch.einsum('gsk,skc->gsc', v, CovB)          # (G,S,C)
        u = t - diagB.unsqueeze(0) * v                     # (G,S,C)
        CovTerm = (2.0 * v * u).permute(0, 2, 1)           # (G,C,S)

    g_Var += CovTerm

    # a, b, grad_PY (unchanged)
    a = (g_Var - 2.0 * g_Exp / Exp.unsqueeze(1) * Var.unsqueeze(1)) / (Exp.unsqueeze(1).pow(2))
    Diff = Y - torch.log(Exp) - Var / (2.0 * Exp.square())
    b = -Diff.unsqueeze(1) * (2.0 * g_Exp / Exp.unsqueeze(1) + a)
    scaling = 0.5 / SigmaY.square().unsqueeze(1).clamp_min(EPS)
    grad_PY = -(scaling * (a + b)).permute(2, 0, 1)

    return grad_PX * (1.0 / weight) + grad_PY


def grad_Omega_C(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0,
                      Ngene, Ncell, Nsample, weight):
    # ----- PX term -----
    AlphaN = Alpha0 + 0.5 * Nsample
    NuExp  = torch.sum(Nu, dim=0) / Nsample
    ExpBetaN = Beta0 + (Nsample - 1) * 0.5 * Omega.square() \
             + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (Omega.square()/Nsample + (NuExp - Mu0).square())
    ExpBetaN = ExpBetaN + 0.5 * torch.sum((Nu - NuExp).square(), dim=0)
    Nominator = -AlphaN * (Nsample - 1) * Omega + Kappa0/(Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN.clamp_min(EPS)

    # ----- PY term (memory-safe) -----
    B0     = Beta.sum(dim=1).clamp_min(EPS)                        # (S,)
    Btilda = Beta / B0.unsqueeze(1)                                 # (S,C)

    Exp = ExpQ_C(Nu, Beta, Omega).clamp_min(EPS)                    # (G,S)
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample).clamp_min(0.0)  # (G,S)

    # CovB and its diagonal
    CovB  = -torch.einsum('sl,sk->slk', Btilda, Btilda) / (1.0 + B0).unsqueeze(1).unsqueeze(2)  # (S,C,C)
    diagB = CovB.diagonal(dim1=1, dim2=2)                                                       # (S,C)

    # v = E[X] with guard
    argx = torch.clamp(Nu + 0.5 * Omega.square().unsqueeze(0), max=EXP_MAX)  # (S,G,C)
    ExpX = torch.exp(argx)                                                    # (S,G,C)

    # g_Exp includes Ω factor
    g_Exp = (ExpX * Btilda.unsqueeze(1) * Omega.unsqueeze(0)).permute(1, 2, 0)  # (G,C,S)

    # Var terms (guard exponents)
    VarX = torch.exp(torch.clamp(2*Nu + 2*Omega.square().unsqueeze(0), max=EXP_MAX))  # (S,G,C)
    VarB = Btilda * (1.0 - Btilda)
    VarB /= (B0 + 1.0).unsqueeze(1)

    # base pieces
    first_term  = 2.0 * VarX * (VarB + Btilda.square()).unsqueeze(1)   # (S,G,C)
    diag_CovX   = ExpX.square()                                        # (S,G,C)
    second_term = 2.0 * diag_CovX * Btilda.square().unsqueeze(1)       # (S,G,C)

    # >>> the missing Ω factors (match your heavy version’s algebra) <<<
    first_term = (2.0 * Omega.unsqueeze(2)) * first_term.permute(1, 2, 0)   # (G,C,S)
    second_term = (    Omega.unsqueeze(2)) * second_term.permute(1, 2, 0)   # (G,C,S)
    g_Var = first_term - second_term                                        # (G,C,S)

    if _use_cpu_fastpath(Nu):
        v_sgc = ExpX
        t_sgc = torch.bmm(v_sgc, CovB)
        u_sgc = t_sgc - diagB.unsqueeze(1) * v_sgc
        g_Var += (2.0 * v_sgc * u_sgc).permute(1, 2, 0) * Omega.unsqueeze(2)
    else:
        v = ExpX.permute(1, 0, 2)
        t = torch.einsum('gsk,skc->gsc', v, CovB)
        u = t - diagB.unsqueeze(0) * v
        g_Var += (2.0 * v * u).permute(0, 2, 1) * Omega.unsqueeze(2)

    # a, b, grad_PY (unchanged)
    a = (g_Var - 2.0 * g_Exp * Var.unsqueeze(1) / Exp.unsqueeze(1)) / (Exp.square().unsqueeze(1))
    Diff = Y - torch.log(Exp) - Var / (2.0 * Exp.square())
    b = -Diff.unsqueeze(1) * (2.0 * g_Exp / Exp.unsqueeze(1) + a)
    grad_PY = torch.sum(-0.5 / SigmaY.square().unsqueeze(1).clamp_min(EPS) * (a + b), dim=2)

    grad_QX = - Nsample / Omega.clamp_min(EPS)
    return grad_PX * (1.0 / weight) + grad_PY - grad_QX * (1.0 / weight)


def g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    """
    Same output/shape as your g_Exp_Beta_C, but guards exponents and divides.
    Returns: (Nsample, Ncell, Ngene)
    """
    device = Nu.device
    # v = E[X] = exp(Nu + 0.5*Omega^2)
    arg = torch.clamp(Nu + 0.5 * Omega.square().unsqueeze(0), max=EXP_MAX)  # (S,G,C)
    v   = torch.exp(arg)

    B0_safe   = B0.clamp_min(EPS)                          # (S,)
    B0_sq_inv = (1.0 / B0_safe.square()).unsqueeze(1)      # (S,1)

    # (S,C) -> (S,1,C) x (S,C,G) -> (S,1,G) -> (S,G)
    B0mat = torch.matmul((Beta * B0_sq_inv).unsqueeze(1), v.transpose(1, 2)).squeeze(1)  # (S,G)

    # g_Exp = v / B0 - (β/B0^2)·v   ; shape -> (S,C,G)
    g_Exp = (v / B0_safe.unsqueeze(1).unsqueeze(2)) - B0mat.unsqueeze(2)
    return g_Exp.permute(0, 2, 1)   # (S,C,G) -> (S,C,G) matches (Nsample, Ncell, Ngene)


def g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    """
    Memory-safe version (no 4-D CovX). Matches your output shape (Nsample, Ncell, Ngene).
    """
    S, G, C = Nu.shape
    device = Nu.device
    dtype  = Nu.dtype

    B0_safe = B0.clamp_min(EPS)        # (S,)
    B0Rep   = B0_safe.unsqueeze(1)     # (S,1)

    # Exponentials with overflow guards
    ExpX2 = torch.exp(torch.clamp(2*Nu + 2*Omega.square().unsqueeze(0), max=EXP_MAX))      # (S,G,C)
    ExpX  = torch.exp(torch.clamp(2*Nu +     Omega.square().unsqueeze(0), max=EXP_MAX))    # (S,G,C)

    # ---- coefficients (S,C) ----
    aa     = (B0Rep - Beta) * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aa     = aa / (B0Rep.pow(3) * (B0Rep + 1).square()) + 2 * Beta * (B0Rep - Beta) / B0Rep.pow(3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (B0Rep.pow(3) * (B0Rep + 1).square()) + 2 * Beta * (-Beta) / B0Rep.pow(3)

    # base: transpose(1,2) -> (S,C,G)
    g_Var = ExpX2.transpose(1, 2) * aa.unsqueeze(2)                    # (S,C,G)
    total = (ExpX2 * aaNotT.view(S, 1, C)).sum(dim=2)                  # (S,G)
    g_Var = g_Var + (total.unsqueeze(1) - ExpX2.transpose(1, 2) * aaNotT.unsqueeze(2))

    # - 2 * ExpX * (Beta/B0^2)
    B_B02 = Beta / B0Rep.square()                                      # (S,C)
    g_Var = g_Var - 2.0 * ExpX.transpose(1, 2) * B_B02.unsqueeze(2)    # (S,C,G)

    # + 2 * sum_j (Beta_j^2/B0^3 * ExpX[:,:,j])
    B2_B03 = Beta.square() / B0Rep.pow(3)                              # (S,C)
    Dot    = (B2_B03.unsqueeze(1) * ExpX).sum(dim=2)                   # (S,G)
    g_Var  = g_Var + 2.0 * Dot.unsqueeze(1)                            # (S,C,G)

    # ======= REPLACEMENT FOR COVX TERMS (no (C,C) tensors) =======
    # v = E[X] = exp(Nu + 0.5*Omega^2)
    v      = torch.exp(torch.clamp(Nu + 0.5*Omega.square().unsqueeze(0), max=EXP_MAX))  # (S,G,C)
    v_sum  = v.sum(dim=2)                                                                  # (S,G)
    v_sqsum= (v*v).sum(dim=2)                                                              # (S,G)

    # CovTerm1: from grad of CovB ~ (3*B0+2)/(B0^3 (B0+1)^2) * (ββᵀ)
    scalar      = (3*B0_safe + 2) / (B0_safe.pow(3) * (B0_safe + 1).square())             # (S,)
    beta_dot_v  = torch.einsum('sc,sgc->sg', Beta, v)                                      # (S,G)
    total1      = scalar.unsqueeze(1) * beta_dot_v.square()                                # (S,G)
    diag1       = scalar.unsqueeze(1) * torch.einsum('sc,sgc->sg', Beta.square(), v.square())
    CovTerm1    = (total1 - diag1).unsqueeze(1).expand(S, C, G)                            # (S,C,G)

    # CovTerm2: -2 * Σ_{k≠l} [β_l (B0+1)/(B0(B0+1))^2 * v_l v_k]  -> use sum(v) - v_l
    B0B0_1 = B0Rep * (B0Rep + 1)                                                           # (S,1)
    coeff  = Beta * (B0Rep + 1) / (B0B0_1.square().clamp_min(EPS))                         # (S,C)
    vT     = v.transpose(1, 2)                                                              # (S,C,G)
    CovTerm2 = coeff.unsqueeze(2) * (vT * (v_sum.unsqueeze(1) - vT))                        # (S,C,G)

    g_Var = g_Var + CovTerm1 - 2.0 * CovTerm2
    return g_Var


def g_PY_Beta_C(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):
    """
    Uses the safe g_Exp_Beta and g_Var_Beta plus guarded Exp/Var.
    Returns: (Nsample, Ncell)
    """
    # Exp, Var with guards
    Exp = ExpQ_C(Nu, Beta, Omega).clamp_min(EPS)                                  # (G,S)
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample).clamp_min(0.0)           # (G,S)

    # grads wrt Beta
    g_Exp = g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)         # (S,C,G)
    g_Var = g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)         # (S,C,G)

    # a term
    Exp_t = Exp.permute(1, 0)                                                     # (S,G)
    Var_t = Var.permute(1, 0)                                                     # (S,G)
    a = (g_Var * Exp_t.unsqueeze(1) - 2.0 * g_Exp * Var_t.unsqueeze(1)) / Exp_t.unsqueeze(1).pow(3)

    # b term
    varExp2_t = (Var / (2.0 * Exp.square())).permute(1, 0)                        # (S,G)
    diff = (Y.permute(1, 0).unsqueeze(1) - torch.log(Exp).permute(1, 0).unsqueeze(1) - varExp2_t.unsqueeze(1))
    b = - diff * (2.0 * g_Exp / Exp_t.unsqueeze(1) + a)

    # combine, weight, sum over gene
    SigmaY_sq_t = SigmaY.square().permute(1, 0).unsqueeze(1).clamp_min(EPS)       # (S,1,G)
    grad_PY = - torch.sum(0.5 / SigmaY_sq_t * (a + b), dim=2)                     # (S,C)
    return grad_PY

from dataclasses import dataclass, field
import time

@dataclass
class OptLogger:
    label: str = "run"
    records: list = field(default_factory=list)
    _t0: float | None = None
    _closure_calls: int = 0

    def start(self):
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def bump_closure(self, n: int = 1):
        self._closure_calls += n

    def push(self, *, outer_step: int, phase: str, obj_val: float,
             grad_norms: dict | None = None, note: str = ""):
        if self._t0 is None:
            self.start()
        grad_norms = grad_norms or {}
        self.records.append({
            "t": time.perf_counter() - self._t0,   # seconds since first push
            "step": int(outer_step),
            "phase": str(phase),
            "obj": float(obj_val),                 # ELBO
            "gNu": float(grad_norms.get("Nu", 0.0)),
            "gOm": float(grad_norms.get("Omega", 0.0)),
            "gBe": float(grad_norms.get("Beta", 0.0)),
            "closures": int(self._closure_calls),
            "note": note,
        })


class BLADE:
    _cuda_message_printed = False

    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,
                 Alpha0=1, Beta0=1, Kappa0=1,
                 Nu_Init=None, Omega_Init=1, Beta_Init=None,
                 fix_Beta=False, fix_Nu=False, fix_Omega=False,
                 device=None):
        # 1) Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        if not BLADE._cuda_message_printed:
            print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for computation.")
            BLADE._cuda_message_printed = True

        # 2) Keep a weight tensor (we'll convert dtype later in one go)
        self.weight = torch.tensor(1, device=self.device)

        # 3) Core tensors (initially as torch tensors; dtype normalized later)
        self.Y = torch.as_tensor(Y, device=self.device)

        # dims
        self.Ngene, self.Nsample = self.Y.shape

        # fixed/flags
        self.Fix_par = {'Beta': fix_Beta, 'Nu': fix_Nu, 'Omega': fix_Omega}

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

        # --- NEW: normalize dtype to float32 (much faster on GPU) ---
        for name in ["Y", "Mu0", "SigmaY", "Alpha", "Alpha0", "Beta0", "Kappa0", "Omega", "Nu", "Beta", "weight"]:
            t = getattr(self, name)
            setattr(self, name, t.to(self.device, dtype=torch.float32))

            # 4) Enable fast matmul on Ampere/Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # 5) (Optional) autocast for forward-only parts
        self._use_amp = torch.cuda.is_available()

                
                # 6) Use direct python/tensor ops (no torch.compile to avoid thread/TLS issues)
        # direct function (already fine as-is)
        self._ExpF = ExpF_C

        # same as your lambda but cleaner (no capture, picklable)
        self._ExpQ = ExpQ_C

        # needs fixed sizes
        self._VarQ = partial(
            VarQ_C,
            Ngene=self.Ngene, Ncell=self.Ncell, Nsample=self.Nsample
        )

        # bake in constants, leave (Nu, Omega) free
        self._EstepPX = partial(
            Estep_PX_C,
            self.Mu0,
            Alpha0=self.Alpha0, Beta0=self.Beta0, Kappa0=self.Kappa0,
            Ncell=self.Ncell, Nsample=self.Nsample
        )

        # bake in constants, leave (Nu, Omega, Beta) free
        self._EstepPY = partial(
            Estep_PY_C,
            self.Y, self.SigmaY,
            Ngene=self.Ngene, Ncell=self.Ncell, Nsample=self.Nsample
        )

        self._compiled_ok = False



        # --- NEW: make trainable tensors Parameters so optimizers can update them ---
        # Respect Fix_par: if fixed, still wrap as Parameter but with requires_grad=False.
        self.Nu    = torch.nn.Parameter(self.Nu,    requires_grad=not self.Fix_par['Nu'])
        self.Omega = torch.nn.Parameter(self.Omega, requires_grad=not self.Fix_par['Omega'])
        self.Beta  = torch.nn.Parameter(self.Beta,  requires_grad=not self.Fix_par['Beta'])

        # near the end of __init__
        self.use_compile = False  # start False; enable only for Adam/SGD warm-up

        torch.set_float32_matmul_precision("high")
        if self.use_compile:
            try:
                self.E_step = torch.compile(self.E_step, mode="reduce-overhead")
            except Exception:
                pass


    def to_device(self, device):
        self.device = torch.device(device)

        # Move all tensor attributes to the new device and enforce float32
        self.Y = self.Y.to(dtype=torch.float32, device=self.device)
        self.Mu0 = self.Mu0.to(dtype=torch.float32, device=self.device)
        self.SigmaY = self.SigmaY.to(dtype=torch.float32, device=self.device)
        self.Alpha = self.Alpha.to(dtype=torch.float32, device=self.device)
        self.Omega = self.Omega.to(dtype=torch.float32, device=self.device)
        self.Nu = self.Nu.to(dtype=torch.float32, device=self.device)
        self.Beta = self.Beta.to(dtype=torch.float32, device=self.device)
        self.Alpha0 = self.Alpha0.to(dtype=torch.float32, device=self.device)
        self.Beta0 = self.Beta0.to(dtype=torch.float32, device=self.device)
        self.Kappa0 = self.Kappa0.to(dtype=torch.float32, device=self.device)

            
    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = torch.matmul(torch.exp(Nu), F.T)
        return torch.sum(torch.square(self.Y - Ypred))

    def ExpF(self, Beta):
        #NSample by Ncell (Expectation of F)
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

    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        # First term: negative sum of log-gamma of Alpha minus log-gamma of sum(Alpha)
        term1 = -(torch.sum(torch.special.gammaln(self.Alpha)) - 
                torch.sum(torch.special.gammaln(torch.sum(self.Alpha, dim=1))))
        
        # Second term: sum of (Alpha-1) * (digamma(Beta) - digamma(sum(Beta)))
        digamma_Beta = torch.special.digamma(Beta)
        
        # Expand digamma(sum(Beta)) to match Beta's shape through tiling
        digamma_sum_Beta = torch.special.digamma(torch.sum(Beta, dim=1))  # Shape: (Nsample,)
        digamma_sum_Beta_expanded = digamma_sum_Beta.unsqueeze(1)  # Shape: (Nsample, 1)
        digamma_sum_Beta_tiled = digamma_sum_Beta_expanded.expand(-1, self.Ncell)  # Shape: (Nsample, Ncell)
        
        term2 = torch.sum((self.Alpha - 1) * (digamma_Beta - digamma_sum_Beta_tiled))
        
        return term1 + term2


    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample * torch.sum(torch.log(Omega))

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

    def grad_Nu(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Nu_C(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_C(self.Y, self.SigmaY, Nu, Omega, Beta,
                          self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta_C(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

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


    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1/self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        QX = self.Estep_QX(Omega) * (1/self.weight)
        QF = self.Estep_QF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        return PX+PY+PF-QX-QF

    def _finite_clamp_(self):
        """Project to SciPy L-BFGS-B–equivalent box constraints and scrub non-finite.
        Match old numpy optimizer: Nu unbounded; Omega,Beta in [1e-7, 100].
        """
        with torch.no_grad():
            # 1) Scrub non-finite entries
            for p in (self.Nu, self.Omega, self.Beta):
                if isinstance(p, torch.Tensor):
                    bad = ~torch.isfinite(p)
                    if bad.any():
                        # Replace by a safe neutral value:
                        # - Nu: 0 (old code allowed any real; 0 keeps exp stable)
                        # - Omega/Beta: mid of the box (or 1.0) then clamped below
                        fill = 0.0 if p is self.Nu else 1.0
                        p[bad] = fill

            # 2) Enforce box bounds exactly as before (only on Omega and Beta)
            if isinstance(self.Omega, torch.Tensor):
                self.Omega.clamp_(min=1e-7, max=100.0)
            if isinstance(self.Beta,  torch.Tensor):
                self.Beta.clamp_(min=1e-7, max=100.0)

            # 3) Do NOT clamp Nu (unbounded in the old SciPy version)
            #    (We rely on internal exp-guards like EXP_MAX inside math kernels.)


    def _analytical_grads_(self):
        """Fill .grad with negative ascent direction (PyTorch optimizers minimize)."""
        with torch.no_grad():
            if not self.Fix_par['Nu']:
                self.Nu.grad = -self.grad_Nu(self.Nu, self.Omega, self.Beta)
            if not self.Fix_par['Omega']:
                self.Omega.grad = -self.grad_Omega(self.Nu, self.Omega, self.Beta)
            if not self.Fix_par['Beta']:
                self.Beta.grad = -self.grad_Beta(self.Nu, self.Omega, self.Beta)

    def Optimize(self, steps=60, lr=2e-2, method="lbfgs", grad_clip=1e4, **opt_kwargs):
        """
        GPU-native optimizer using analytical gradients (no autograd graph).
        Maximizes E_step. Enforces finite/clamped params (outside closure).
        E_step stays FP32; gradients can use bf16 AMP on GPU.
        Auto-recovers from LBFGS line-search failures with a short Adam warm-up.
        Extra kwargs are forwarded to the chosen optimizer.

        Logging:
        - pass logger=OptLogger(...), phase="adam"/"lbfgs", outer_step=<int>
        - one record is appended per outer step
        """
        import contextlib
        logger: "OptLogger | None" = opt_kwargs.pop("logger", None)
        phase  = opt_kwargs.pop("phase", None) or method.lower()
        outer_step = int(opt_kwargs.pop("outer_step", -1))

        # -------- AMP for GRADIENTS ONLY (not for E_step) --------
        def _amp_autocast_grad(enabled: bool = True, dtype=torch.bfloat16):
            if not enabled or not torch.cuda.is_available():
                return contextlib.nullcontext()
            try:
                major, _ = torch.cuda.get_device_capability()
            except Exception:
                major = 0
            if dtype is torch.bfloat16 and major < 8:  # Ampere/Hopper+
                return contextlib.nullcontext()
            try:
                return torch.amp.autocast('cuda', dtype=dtype)
            except AttributeError:
                return torch.cuda.amp.autocast(dtype=dtype)

        amp_grads = bool(opt_kwargs.pop("amp_grads", True))
        amp_ctx   = _amp_autocast_grad(enabled=amp_grads, dtype=torch.bfloat16)

        # --- 0) make sure starting point is sane (once) ---
        self._finite_clamp_()

        # --- 1) prepare trainables respecting Fix_par ---
        trainable = []
        for name in ('Nu', 'Omega', 'Beta'):
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

        # --- 2) choose optimizer (thread through **opt_kwargs) ---
        method_l = method.lower()
        lbfgs_allowed = {
            "lr","max_iter","max_eval","tolerance_grad","tolerance_change",
            "history_size","line_search_fn"
        }
        adam_allowed = {
            "betas","eps","weight_decay","amsgrad","capturable",
            "foreach","maximize","differentiable","fused","lr"
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

        best_obj = None
        patience, wait = 25, 0

        # --- helper: scrub non-finite grads; clip only when using Adam ---
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
            if not self.Fix_par['Nu']   and getattr(self.Nu,   "grad", None) is not None:   d["Nu"]    = self.Nu.grad.norm().item()
            if not self.Fix_par['Omega'] and getattr(self.Omega,"grad", None) is not None: d["Omega"] = self.Omega.grad.norm().item()
            if not self.Fix_par['Beta'] and getattr(self.Beta, "grad", None) is not None:  d["Beta"]  = self.Beta.grad.norm().item()
            return d

        # --- 3) LBFGS closure (analytical grads; E_step FP32; no clamp here) ---
        last_grad_norms = {}
        def closure():
            if logger: logger.bump_closure()
            opt.zero_grad(set_to_none=True)

            obj = self.E_step(self.Nu, self.Beta, self.Omega)
            if not torch.isfinite(obj):
                self._finite_clamp_()
                return torch.tensor(1e30, device=self.device, dtype=torch.float32)

            with amp_ctx:
                self._analytical_grads_()
            _clean_grads(do_clip=False)

            nonlocal last_grad_norms
            last_grad_norms = _grad_norms_dict()
            return -obj  # LBFGS minimizes

        # --- 4) main loop ---
        if logger: logger.start()
        for _ in range(steps):
            if use_lbfgs:
                with torch.no_grad():
                    snapshot = [p.clone() for p in trainable]
                try:
                    with maybe_disable_dynamo():
                        loss = opt.step(closure)
                    obj_val = (-loss).item()
                    self._finite_clamp_()
                except (IndexError, RuntimeError):
                    with torch.no_grad():
                        for p, s in zip(trainable, snapshot):
                            p.copy_(s)
                    warm = torch.optim.Adam(trainable, lr=max(adam_kwargs.get("lr", lr) * 0.5, 1e-3))
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

            # ---- LOG ONE RECORD PER OUTER STEP ----
            if logger:
                logger.push(
                    outer_step=outer_step if outer_step >= 0 else 0,
                    phase=phase,
                    obj_val=obj_val,
                    grad_norms=last_grad_norms
                )

            # early stopping (patience)
            if best_obj is None or obj_val > best_obj + 1e-7:
                best_obj, wait = obj_val, 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.log = True
        return self

    # Reestimation of Nu at specific weight
    def Reestimate_Nu(self,weight=100):
        self.weight=weight
        self.Optimize()
        return self

    def Check_health(self):
        # check if optimization is done
        if not hasattr(self, 'log'):
            warnings.warn("No optimization is not done yet", Warning, stacklevel=2)

        # check values in hyperparameters
        if not np.all(np.isfinite(self.Y.cpu().numpy())):
            warnings.warn('non-finite values detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Y.cpu().numpy() < 0):
            warnings.warn('Negative expression levels were detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Alpha.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Alpha', Warning, stacklevel=2)

        if np.any(self.Beta.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Beta', Warning, stacklevel=2)

        if np.any(self.Alpha0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Alpha0', Warning, stacklevel=2)

        if np.any(self.Beta0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Beta0', Warning, stacklevel=2)

        if np.any(self.Kappa0.cpu().numpy() <= 0):
            warnings.warn('Zero or negative values in Kappa0', Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta).cpu.numpy()
        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = np.identity(Expected['Expectation'].shape[1])
                Expected = Expected['Expectation']
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                IndG = np.where(~np.isnan(Expected[sample,:]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group,:] == 1)[0]
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] / np.sum(Fraction[sample,IndCell])  # make fraction sum to one for the group
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] * Expected[sample, group]  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[sample, IndNan] = Fraction[sample, IndNan] / np.sum(Fraction[sample, IndNan])  # normalize the rest of cell types (sum to one)
                Fraction[sample, IndNan] = Fraction[sample, IndNan] * (1-np.sum(Expected[sample, IndG]))  # assign determined fraction for the rest of cell types

        if Temperature is not None:
            self.Alpha = Temperature * Fraction
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = Fraction[sample,:] * np.sum(self.Beta[sample,:])
        self.Alpha = torch.tensor(self.Alpha, device=self.device)



    def Update_Alpha_Group(self, Expected=None, Temperature=None):
        """
        Update Dirichlet prior α using group expectations.
        If Expected is None -> do nothing (keep current α).
        """
        # --- no Expected: no update (prevents unintended pooling across samples) ---
        if Expected is None:
            return  # keep self.Alpha as-is

        # --- Expected provided: original group logic ---
        if isinstance(Expected, dict):
            Group = Expected.get('Group', torch.eye(Expected['Expectation'].shape[1], device=self.device))
            Expected = Expected['Expectation']
        else:
            Group = torch.eye(Expected.shape[1], device=self.device)

        if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
            raise ValueError('Pre-determined fraction is in wrong shape (Nsample × Ncelltype)')

        Expected = torch.as_tensor(Expected, device=self.device, dtype=torch.float32)
        Group    = torch.as_tensor(Group,    device=self.device, dtype=torch.float32)

        AvgBeta = torch.mean(self.Beta, dim=0)
        Fraction_Avg = AvgBeta / torch.sum(AvgBeta)

        for sample in range(self.Nsample):
            Fraction = Fraction_Avg.clone()
            IndG = torch.where(~torch.isnan(Expected[sample]))[0]

            IndCells = []
            for group in IndG:
                IndCell = torch.where(Group[group, :] == 1)[0]
                s = torch.sum(Fraction[IndCell])
                if s > 0:
                    Fraction[IndCell] = Fraction[IndCell] / s
                Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]
                IndCells.extend(IndCell.tolist())

            all_indices = torch.arange(Group.shape[1], device=self.device)
            mask = torch.ones(Group.shape[1], dtype=torch.bool, device=self.device)
            mask[IndCells] = False
            IndNan = all_indices[mask]

            remaining_mass = 1.0 - torch.sum(Expected[sample, IndG])
            if torch.sum(Fraction[IndNan]) > 0:
                Fraction[IndNan] = Fraction[IndNan] / torch.sum(Fraction[IndNan])
            Fraction[IndNan] = Fraction[IndNan] * remaining_mass

            AlphaSum = torch.sum(AvgBeta[IndNan]) / torch.clamp(torch.sum(Fraction[IndNan]), min=1e-12)
            self.Alpha[sample] = Fraction * AlphaSum


    def Update_Alpha_Group_old(self, Expected=None, Temperature=None):  # if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = torch.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / torch.sum(AvgBeta)
        #print("Initial Fraction_Avg:", Fraction_Avg)

        if Expected is not None:  # Reflect the expected values
            # Expectation can be a dictionary (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = torch.eye(Expected['Expectation'].shape[1], device=self.device)
                Expected = Expected['Expectation']
            else:
                Group = torch.eye(Expected.shape[1], device=self.device)

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')
            Expected = torch.tensor(Expected, device=self.device)
            #print("Expected values:\n", Expected)
            #print("Group matrix:\n", Group)

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = Fraction_Avg.clone()
                #print(f"\nSample {sample}:")
                #print("  Initial Fraction:", Fraction)

                # Get indices of expected cell types (non-NaN)
                IndG = torch.where(~torch.isnan(Expected[sample, :]))[0]
                #print("  IndG (indices with expectations):", IndG)

                IndCells = []
                for group in IndG:
                    IndCell = torch.where(Group[group, :] == 1)[0]
                    #print("    Group:", group.item(), "-> IndCell:", IndCell)
                    # Normalize fractions in the group to sum to 1
                    Fraction[IndCell] = Fraction[IndCell] / torch.sum(Fraction[IndCell])
                    #print("    Normalized Fraction for group:", Fraction[IndCell])
                    # Multiply by the expected value for that group
                    Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]
                    #print("    Adjusted Fraction for group:", Fraction[IndCell])
                    IndCells.extend(IndCell.tolist())

                IndNan = torch.tensor(list(set(range(Group.shape[1])) - set(IndCells)), device=Fraction.device)
                #print("  IndNan (cell types with no expectation):", IndNan)
                Fraction[IndNan] = Fraction[IndNan] / torch.sum(Fraction[IndNan])
                Fraction[IndNan] = Fraction[IndNan] * (1 - torch.sum(Expected[sample, IndG]))
                #print("  Adjusted Fraction for non-specified cells:", Fraction[IndNan])

                AlphaSum = torch.sum(AvgBeta[IndNan]) / torch.sum(Fraction[IndNan])
                #print("  AlphaSum:", AlphaSum)
                self.Alpha[sample, :] = Fraction * AlphaSum
                #print("  Updated Alpha for sample:", self.Alpha[sample, :])
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample, :] = AvgBeta



    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_C(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_C(self.Nu, self.Beta, self.Omega)

        a = Var / Exp / Exp
        b = torch.square((self.Y - torch.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = torch.sqrt(a+b)
        else:  # shared in all samples
            self.SigmaY = torch.mean(torch.sqrt(a + b), dim=1, keepdim=True).expand(-1, self.Nsample)



def Optimize(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction):
    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) * 0.1 + t(Init_Fraction) * 10
    obs = BLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
            Nu_Init, Omega_Init, Beta_Init, fix_Nu=True, fix_Omega=True)
    obs.Optimize()
    obs.Fix_par['Nu'] = False
    obs.Fix_par['Omega'] = False
    obs.Optimize()
    return obs


def NuSVR_job(X, Y, Nus, sample):
    X = np.exp(X) - 1
    sols = [NuSVR(kernel='linear', nu=nu).fit(X,Y[:, sample]) for nu in Nus]
    RMSE = [mse(sol.predict(X), Y[:, sample]) for sol in sols]
    return sols[np.argmin(RMSE)]


def SVR_Initialization(X, Y, Nus, Njob=1, fsel=0):
    Ngene, Nsample = Y.shape
    Ngene, Ncell = X.shape
    SVRcoef = np.zeros((Ncell, Nsample))
    Selcoef = np.zeros((Ngene, Nsample))

    with parallel_backend('loky', n_jobs=Njob):
        sols = Parallel(n_jobs=Njob, verbose=10)(
                delayed(NuSVR_job)(X, Y, Nus, i)
                for i in range(Nsample)
                )

    for i in range(Nsample):
        Selcoef[sols[i].support_,i] = 1
        SVRcoef[:,i] = np.maximum(sols[i].coef_,0)

    Init_Fraction = SVRcoef
    for i in range(Nsample):
        Init_Fraction[:,i] = Init_Fraction[:,i]/np.sum(SVRcoef[:,i])

    if fsel > 0:
        Ind_use = Selcoef.sum(1) > Nsample * fsel
        print( "SVM selected " + str(Ind_use.sum()) + ' genes out of ' + str(len(Ind_use)) + ' genes')
    else:
        # print("No feature filtering is done (fsel = 0)")
        Ind_use = np.ones((Ngene)) > 0

    return Init_Fraction, Ind_use


# ---- parallel runtime helpers ----
# ---- parallel runtime helpers (patched) ----
def _set_torch_threads(num_threads: int, interop_threads: int | None = None, *, best_effort: bool = True):
    import os, torch
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
        return torch.cuda.is_available()
    return str(dev_str).startswith("cuda")


def Iterative_Optimization(
    X, stdX, Y, Alpha, Alpha0, Kappa0, SY, Rep, Init_Fraction, Init_Trust=10,
    Expected=None, iter=100, minDiff=1e-4, TempRange=None, Update_SigmaY=False,
    device=None, *, warm_start: bool = True,
    adam_params: dict = None,           # safe defaults handled below
    lbfgs_params: dict = None,
    runtime_threads: int | None = None  # NEW: per-process CPU threads; GPU workers force 1
):
       # --- per-worker thread policy (NO interop here) ---
    if _is_cuda_device(device):
        _set_torch_threads(1, interop_threads=None)  # GPU worker: minimal CPU threads
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))
    else:
        if runtime_threads is not None:
            _set_torch_threads(runtime_threads, interop_threads=None)  # CPU worker: set only intra-op


    # --- defaults for optimizer kwargs (avoid None.get(...) errors) ---
    adam_params = adam_params or {}
    lbfgs_params = lbfgs_params or {}

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

    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) + t(Init_Fraction) * Init_Trust

    obj = BLADE(
        logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0,
        Nu_Init, Omega_Init, Beta_Init,
        device=device
    )

    # --- attach a run logger ---
    run_log = OptLogger(label=f"rep{Rep}_warm{warm_start}")

    obj_func = [None] * iter
    with torch.no_grad():
        obj_func[0] = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))

    for i in range(1, iter):
        if i == 1 and warm_start:
            obj.Optimize(
                method="adam",
                steps=adam_params.get("steps", 20),
                lr=adam_params.get("lr", 5e-3),
                betas=adam_params.get("betas", (0.9, 0.999)),
                grad_clip=adam_params.get("grad_clip", 1e4),
                logger=run_log, phase="adam", outer_step=i
            )
        else:
            try:
                obj.Optimize(
                    method="lbfgs",
                    steps=lbfgs_params.get("steps", 12),
                    lr=lbfgs_params.get("lr", 0.5),
                    max_iter=lbfgs_params.get("max_iter", 12),
                    history_size=lbfgs_params.get("history_size", 50),
                    line_search_fn=lbfgs_params.get("line_search_fn", "strong_wolfe"),
                    grad_clip=lbfgs_params.get("grad_clip", None),
                    logger=run_log, phase="lbfgs", outer_step=i
                )
            except Exception as e:
                print(f"[WARN] LBFGS failed at iter {i} rep {Rep} ({e}), retrying with lr=0.1]")
                obj.Optimize(
                    method="lbfgs",
                    steps=lbfgs_params.get("retry_steps", 8),
                    lr=lbfgs_params.get("retry_lr", 0.1),
                    max_iter=lbfgs_params.get("retry_max_iter", 8),
                    history_size=lbfgs_params.get("retry_history_size", 10),
                    line_search_fn="strong_wolfe",
                    grad_clip=None,
                    logger=run_log, phase="lbfgs-retry", outer_step=i
                )

        if Expected is not None:
            obj.Update_Alpha_Group(Expected=Expected)
        if Update_SigmaY:
            obj.Update_SigmaY()

        with torch.no_grad():
            obj_val = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
            obj_func[i] = obj_val
            if not np.isfinite(obj_val):
                print(f"[WARN] non-finite ELBO at outer iter {i} rep {Rep}; stopping.")
                obj_func = obj_func[:i+1]
                break
            if abs(obj_func[i] - obj_func[i-1]) < minDiff:
                obj_func = obj_func[:i+1]
                break

    # quick polish (unchanged)
    obj.Fix_par['Nu']=False; obj.Fix_par['Omega']=True; obj.Fix_par['Beta']=True
    obj.Optimize(method="lbfgs", steps=12, lr=0.10, max_iter=20, history_size=100,
                 line_search_fn="strong_wolfe", logger=run_log, phase="polish:Nu", outer_step=iter)

    obj.Fix_par['Nu']=True; obj.Fix_par['Omega']=False; obj.Fix_par['Beta']=True
    obj.Optimize(method="lbfgs", steps=12, lr=0.10, max_iter=20, history_size=100,
                 line_search_fn="strong_wolfe", logger=run_log, phase="polish:Omega", outer_step=iter)

    obj.Fix_par['Nu']=True; obj.Fix_par['Omega']=True; obj.Fix_par['Beta']=False
    obj.Optimize(method="lbfgs", steps=12, lr=0.10, max_iter=20, history_size=100,
                 line_search_fn="strong_wolfe", logger=run_log, phase="polish:Beta", outer_step=iter)

    obj.Fix_par['Nu']=False; obj.Fix_par['Omega']=False; obj.Fix_par['Beta']=False

    with torch.no_grad():
        obj_func.append(float(obj.E_step(obj.Nu, obj.Beta, obj.Omega)))

    obj.train_log = getattr(obj, "train_log", []) + run_log.records
    return obj, obj_func, Rep

def Framework_Iterative(
    X, stdX, Y, Ind_Marker=None,
    Alpha=1, Alpha0=1000, Kappa0=1, sY=1,
    Nrep=10, Njob=None, fsel=0, Update_SigmaY=False, Init_Trust=10,
    Expectation=None, Temperature=None, IterMax=100,
    *, warm_start: bool = True,
    collect_logs: bool = False,
    adam_params: dict = None,
    lbfgs_params: dict = None,
    backend: str = "auto",              # "auto" | "gpu" | "cpu"
    threads_per_job: int | None = None  # per-process CPU threads
):
    # --- sanitize optimizer dicts (Iterative_Optimization uses .get(...)) ---
    adam_params  = adam_params  or {}
    lbfgs_params = lbfgs_params or {}

    args = locals()

    Ngene, Nsample = Y.shape
    if Ind_Marker is None:
        Ind_Marker = [True] * Ngene

    X_small    = X[Ind_Marker, :]
    Y_small    = Y[Ind_Marker, :]
    stdX_small = stdX[Ind_Marker, :]

    # ---------- Decide execution mode & set parent thread limits *first* ----------
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if backend == "auto":
        backend = "gpu" if ngpu > 0 else "cpu"

    if backend == "gpu":
        if ngpu == 0:
            raise RuntimeError("backend='gpu' requested but no CUDA devices are visible.")
        devices = [f"cuda:{i}" for i in range(ngpu)]
        if Njob is None:
            Njob = ngpu
        Njob_eff = min(Njob, ngpu)
        joblib_backend = "loky"
        runtime_threads = 1           # each GPU worker uses minimal CPU threads
        # Parent: best-effort set intra & interop once, before any parallel work
        _set_torch_threads(runtime_threads, interop_threads=1)
        print(f"[Framework] GPU mode: {Njob_eff} workers over {ngpu} GPUs; runtime_threads={runtime_threads}")

    else:  # CPU
        devices = ["cpu"]
        logical_cores = os.cpu_count() or 1
        if Njob is None:
            Njob = min(8, logical_cores)   # sensible default
        if threads_per_job is None:
            threads_per_job = max(1, logical_cores // max(1, Njob))
        Njob_eff = max(1, min(Njob, logical_cores))
        joblib_backend = "loky"
        runtime_threads = threads_per_job
        # Parent: best-effort set intra & interop once, before any parallel work
        _set_torch_threads(runtime_threads, interop_threads=1)
        print(f"[Framework] CPU mode: {Njob_eff} workers × {runtime_threads} threads (≈ {Njob_eff*runtime_threads} total)")

    # ---------- Now safe to do any threaded work in parent (e.g., SVR init) ----------
    print(f"start optimization using marker genes: {Y_small.shape[0]} genes out of {Ngene} genes.")
    print('Initialization with Support vector regression')

    # NOTE: SVR_Initialization uses joblib; parent thread limits are already set.
    Init_Fraction, Ind_use = SVR_Initialization(
        X_small, Y_small, Njob=(Njob or 1), Nus=[0.25, 0.5, 0.75]
    )

    def _iter_one(rep):
        # round-robin assign GPUs (or "cpu")
        dev = devices[rep % len(devices)]
        return Iterative_Optimization(
            X_small[Ind_use, :],
            stdX_small[Ind_use, :],
            Y_small[Ind_use, :],
            Alpha, Alpha0, Kappa0, sY,
            rep, Init_Fraction,
            Expected=Expectation,
            Init_Trust=Init_Trust,
            iter=IterMax,
            Update_SigmaY=Update_SigmaY,
            device=dev,
            warm_start=warm_start,
            adam_params=adam_params,
            lbfgs_params=lbfgs_params,
            runtime_threads=runtime_threads  # worker sets *only* intra-op
        )

    if Temperature is None or Temperature is False:
        # --- parallel runs (no annealing) ---
        with parallel_backend(joblib_backend, n_jobs=Njob_eff):
            triples = Parallel(n_jobs=Njob_eff, verbose=10)(
                delayed(_iter_one)(rep) for rep in range(Nrep)
            )
        outs, convs, Reps = zip(*triples)

        # Evaluate ELBO (no grad, robust to NaN/Inf)
        cri = []
        with torch.no_grad():
            for obj in outs:
                val = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
                val = float(val.detach().cpu().item())
                if not math.isfinite(val):
                    val = float("-inf")
                cri.append(val)

        if all(v == float("-inf") for v in cri):
            raise RuntimeError("All runs produced non-finite ELBO. Try lower lr/steps or inspect inputs.")

        best = int(np.argmax(cri))
        out  = outs[best]
        conv = convs[best]

    else:
        # --- temperature schedule branch (serial here) ---
        if Temperature is True:
            Temperature = [1, 100]
        else:
            if len(Temperature) != 2:
                raise ValueError('Temperature must be None, True, or [Tmin, Tmax].')
            if Temperature[1] < Temperature[0]:
                raise ValueError('Max temperature must be ≥ min temperature.')

        triples = [Iterative_Optimization(
                        X_small[Ind_use, :], stdX_small[Ind_use, :], Y_small[Ind_use, :],
                        Alpha, Alpha0, Kappa0, sY, rep, Init_Fraction,
                        Expected=Expectation, Init_Trust=Init_Trust,
                        TempRange=np.linspace(Temperature[0], Temperature[1], num=IterMax),
                        Update_SigmaY=Update_SigmaY,
                        warm_start=warm_start,
                        adam_params=adam_params,
                        lbfgs_params=lbfgs_params,
                        runtime_threads=runtime_threads
                    ) for rep in range(Nrep)]

        outs, convs, Reps = zip(*triples)
        with torch.no_grad():
            cri = [float(obj.E_step(obj.Nu, obj.Beta, obj.Omega)) for obj in outs]
        best = int(np.nanargmax(cri))
        out  = outs[best]
        conv = convs[best]

    if collect_logs:
        logs = [{"rep": r, "log": getattr(o, "train_log", [])} for o, r in zip(outs, Reps)]
        return out, conv, zip(outs, cri), args, logs

    return out, conv, zip(outs, cri), args













#########NUMBA functions for purification########

@njit(fastmath=True)
def ExpF_numba(Beta, Ncell):
    #NSample by Ncell (Expectation of F)
    output = np.empty(Beta.shape)
    for c in range(Ncell):
        output[:,c] = Beta[:,c]/np.sum(Beta, axis=1)
    return output



@njit(fastmath=True)
def ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Expected value of Y)
    ExpB = ExpF_numba(Beta, Ncell) # Nsample by Ncell
    out = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            out[:,i] = out[:,i] + ExpB[i,c] * np.exp(Nu[i,:,c] + 0.5*np.square(Omega[:,c]))

    return out 


@njit(fastmath=True)
def VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell
    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    # Nsample Ncell Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    # Ngene by Nsample by Ncell by Ncell
    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = np.exp(Nu[i,:,k] + Nu[i,:,l] + \
                        0.5*(np.square(Omega[:,k]) + np.square(Omega[:,l])))

    VarTerm = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            VarTerm[:,i] = VarTerm[:,i] + \
                np.exp(2*Nu[i,:,c] + 2*np.square(Omega)[:,c])*(VarB[i,c] + np.square(Btilda[i,c])) \
                    - np.exp(2*Nu[i,:,c] + np.square(Omega[:,c]))*(np.square(Btilda[i,c]))

    # Ngene by Ncell
    CovTerm = np.zeros((Ngene, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,i] = CovTerm[:,i] + CovX[:,i,l,k] * CovB[i,l,k]
     
    return VarTerm + CovTerm


@njit(fastmath=True)
def Estep_PY_numba(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    a = Var / Exp / Exp
                            
    return np.sum(
            -0.5 / np.square(SigmaY) * (a + np.square((Y-np.log(Exp)) - 0.5 * a))
            )


@njit(fastmath = True)
def Estep_PX_numba(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = np.sum(Nu, 0)/Nsample # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5*Nsample # Posterior Alpha

    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    return np.sum(- AlphaN * np.log(ExpBetaN))

@njit(fastmath=True)
def grad_Nu_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # return Nsample by Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample

    Diff = np.zeros((Ngene, Ncell))
    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)
        Diff = Diff + (Nu[i,:,:] - NuExp) / Nsample

    Nominator = np.empty((Nsample, Ngene, Ncell)) 
    for i in range(Nsample):
        Nominator[i,:,:] = Nu[i,:,:] - NuExp - Diff + Kappa0 / (Kappa0+Nsample) * (NuExp - Mu0)
   
    grad_PX = - AlphaN * Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    ExpX = np.empty(Nu.shape) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.empty(Nu.shape) 
    for i in range(Nsample):
        VarX[i,:,:] = np.exp(2*Nu[i,:,:] + 2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 2*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]/Exp*Var) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])

    grad_PY = np.zeros((Nsample, Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,:,c] = -np.transpose( 0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]))

    return grad_PX *(1/weight) + grad_PY


@njit(fastmath = True)
def grad_Omega_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0)/Nsample
    ExpBetaN = Beta0 + (Nsample-1)/2*np.square(Omega) + \
            Kappa0*Nsample/(2*(Kappa0 + Nsample)) * (np.square(Omega)/Nsample + np.square(NuExp - Mu0))

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5*np.square(Nu[i,:,:] - NuExp)

    Nominator = - AlphaN * (Nsample-1)*Omega + Kappa0 /(Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1) # Nsample
    Btilda = ExpF_numba(Beta, Ncell) # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample) # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:,l,k] = - Btilda[:,l] * Btilda[:,k] / (1+B0)

    ExpX = np.exp(Nu) # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:,i,l,k] = ExpX[i,:,l] * ExpX[i,:,k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:,l,i] = CovTerm[:,l,i] + 2*CovX[:,i,l,k]*CovB[i,l,k]*Omega[:,l]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:,c,i] = ExpX[i,:,c]*Btilda[i,c]*Omega[:,c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.exp(2*Nu)
    for i in range(Nsample):
        VarX[i,:,:] = VarX[i,:,:] * np.exp(2*np.square(Omega))

    VarB = Btilda * (1-Btilda)
    for c in range(Ncell):
        VarB[:,c] = VarB[:,c] / (B0+1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:,c,i] = 4*Omega[:,c]*VarX[i,:,c]*(VarB[i,c] + np.square(Btilda[i,c])) - 2*Omega[:,c]*CovX[:,i,c,c]*np.square(Btilda[i,c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:,c,:] = (g_Var[:,c,:] - 2*g_Exp[:,c,:]*Var/Exp) / np.power(Exp,2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2*np.square(Exp))
    for c in range(Ncell):
        b[:,c,:] = - Diff * (2*g_Exp[:,c,:] / Exp + a[:,c,:])

    grad_PY = np.zeros((Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:,c] = np.sum(-0.5/np.square(SigmaY) * (a[:,c,:] + b[:,c,:]), axis=1)

    # Q(X) (fourth term)
    grad_QX =  - Nsample / Omega

    return grad_PX * (1/weight) + grad_PY - grad_QX * (1/weight)


@njit(fastmath = True)
def g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
    ExpX = np.exp(Nu)
    for i in range(Nsample):
        ExpX[i,:,:] = ExpX[i,:,:]*np.exp(0.5*np.square(Omega)) #Nsample by Ngene by Ncell
    B0mat = np.empty(Beta.shape)
    for c in range(Ncell):
        B0mat[:,c] =Beta[:,c]/np.square(B0)

    tmp = np.empty((Nsample, Ngene))
    tExpX = np.ascontiguousarray(ExpX.transpose(0,2,1)) ## Make tExpX contiguous again
    for i in range(Nsample):
        tmp[i,:] = np.dot(B0mat[i,:], tExpX[i,...])
    B0mat = tmp

    g_Exp = np.empty((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Exp[s,c,:] = t(ExpX[s,:,c] / B0[s]) - B0mat[s,:]

    return g_Exp


@njit(fastmath=True)
def g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    
    B0Rep = np.empty(Beta.shape) # Nsample by Ncell
    for c in range(Ncell):
        B0Rep[:,c] = B0

    aa = (B0Rep - Beta)*B0Rep*(B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa/(np.power(B0Rep,3) * np.square(B0Rep + 1))
    aa = aa + 2*Beta*(B0Rep - Beta)/np.power(B0Rep,3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3*B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (np.power(B0Rep,3) * np.square(B0Rep + 1))
    aaNotT = aaNotT + 2*Beta*(0 - Beta)/np.power(B0Rep,3)
    
    ExpX2 = 2*Nu #Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX2[i,:,:,] = np.exp(ExpX2[i,:,:] + 2*np.square(Omega))

    g_Var = np.zeros((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = t(ExpX2[s,:,c]) * aa[s,c]
  
    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                for s in range(Nsample):
                    g_Var[s,i,:] = g_Var[s,i,:] + t(ExpX2[s,:,j])* aaNotT[s,j]

    B_B02 = Beta / np.square(B0Rep) # Beta / (Beta0^2) / Nsample by Ncell
    B0B0_1 = B0Rep * (B0Rep + 1) # Beta0 (Beta0+1) / Nsample by Nell
    B2_B03 = np.square(Beta) / np.power(B0Rep, 3) # Beta^2 / (Beta0^3) / Nsample by Ncell
        
    ExpX = np.empty(Nu.shape)
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(2*Nu[i,:,:]+np.square(Omega))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s,c,:] = g_Var[s,c,:] - 2 * t(ExpX[s,:,c]) * B_B02[s,c]
    
    Dot = np.zeros((Nsample, Ngene))
    for i in range(Nsample):
        for c in range(Ncell):
            Dot[i,:] = Dot[i,:] + B2_B03[i,c] * ExpX[i,:,c]

    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + 2*Dot
    
    # Ngene by Nsample by Ncell by N cell
    ExpX = np.empty((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        ExpX[i,:,:] = np.exp(Nu[i,:,:] + 0.5*np.square(Omega))
    CovX = np.empty((Nsample, Ngene, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[i,:,l,k] = ExpX[i,:,l] * ExpX[i,:,k]
                
    gradCovB = np.empty((Nsample, Ncell, Ncell))
    B03_2_B03_B0_1 = (3*B0 + 2) / np.power(B0,3) / np.square(B0+1)
    for l in range(Ncell):
        for k in range(Ncell):
            gradCovB[:,l,k] = Beta[:,l] * Beta[:,k] * B03_2_B03_B0_1

    # Nsample by Ncell by Ncell by Ngene
    CovTerm1 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    CovTerm2 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / np.square(B0B0_1) # Nsample by Ncell
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                if l != k:
                    CovTerm1[i,l,k,:] = gradCovB[i,l,k]*CovX[i,:,l,k]
                    CovTerm2[i,l,k,:] = B_B0_1_B0B0_1[i,l]*CovX[i,:,l,k]

    for c in range(Ncell):
        g_Var[:,c,:] = g_Var[:,c,:] + np.sum(np.sum(CovTerm1, axis=1), axis=1)
    g_Var = g_Var - 2*np.sum(CovTerm2, axis=1)

    return g_Var


@njit(fastmath=True)
def g_PY_Beta(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY


@njit(fastmath=True)
def g_PY_Beta_numba(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):

    # Ngene by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell be Ngene
    g_Exp = g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
        
    # Nsample by Ncell by Ngene
    a = np.empty((Nsample, Ncell, Ngene))
    for c in range(Ncell):
        a[:,c,:] = np.divide((g_Var[:,c,:] * t(Exp) - 2 * g_Exp[:,c,:]*t(Var)),np.power(t(Exp),3))
    
    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2*np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s,c,g] = - (Y[g,s] - np.log(Exp[g,s]) - Var_Exp2[g,s]) *(2*np.divide(g_Exp[s,c,g],Exp[g,s]) + a[s,c,g])

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s,c] = grad_PY[s,c] - np.sum(0.5 / np.square(SigmaY[:,s]) * (a[s,c,:] + b[s,c,:]))
    
    return grad_PY


######Casting function#######

def to_torch(array, device='cuda'):
    return torch.tensor(array).to(device)

def convert_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    return tensor

def ensure_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()  # Ensures tensor is on CPU and converts to numpy
    return array

class BLADE_numba:
    def __init__(self, Y, SigmaY=0.05, Mu0=2, Alpha=1,
                 Alpha0=1, Beta0=1, Kappa0=1,
                 Nu_Init=None, Omega_Init=1, Beta_Init=None,
                 fix_Beta=False, fix_Nu=False, fix_Omega=False):
        self.weight = 1

        # Ensure all tensor inputs are converted to numpy arrays
        self.Y = ensure_numpy(Y)
        self.Ngene, self.Nsample = self.Y.shape

        self.Mu0 = ensure_numpy(Mu0) if isinstance(Mu0, np.ndarray) else np.zeros((self.Ngene, Mu0))
        self.Ncell = self.Mu0.shape[1]

        self.SigmaY = ensure_numpy(SigmaY) if isinstance(SigmaY, np.ndarray) else np.ones((self.Ngene, self.Nsample)) * SigmaY
        self.Alpha = ensure_numpy(Alpha) if isinstance(Alpha, np.ndarray) else np.ones((self.Nsample, self.Ncell)) * Alpha
        self.Omega = ensure_numpy(Omega_Init) if isinstance(Omega_Init, np.ndarray) else np.zeros((self.Ngene, self.Ncell)) + Omega_Init
        self.Nu = ensure_numpy(Nu_Init) if Nu_Init is not None else np.zeros((self.Nsample, self.Ngene, self.Ncell))
        self.Beta = ensure_numpy(Beta_Init) if isinstance(Beta_Init, np.ndarray) else np.ones((self.Nsample, self.Ncell))
        self.Alpha0 = ensure_numpy(Alpha0) if isinstance(Alpha0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Alpha0
        self.Beta0 = ensure_numpy(Beta0) if isinstance(Beta0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Beta0
        self.Kappa0 = ensure_numpy(Kappa0) if isinstance(Kappa0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Kappa0

        self.Fix_par = {
            'Beta': fix_Beta,
            'Nu': fix_Nu,
            'Omega': fix_Omega
        }


   
    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = np.dot(np.exp(Nu), t(F))
        return np.sum(np.square(self.Y-Ypred))

    def ExpF_numba(self, Beta):
        #NSample by Ncell (Expectation of F)
        return ExpF_numba(Beta, self.Ncell)

    def ExpQ_numba(self, Nu, Beta, Omega):
        # Ngene by Nsample (Expected value of Y)
        return ExpQ_numba(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    def VarQ_numba(self, Nu, Beta, Omega):
        # Ngene by Nsample (Variance value of Y)
        return VarQ_numba(Nu, Beta, Omega, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(X | mu0, Kappa0, Alpha0, Beta0)
    def Estep_PX(self, Nu, Omega):
        return Estep_PX_numba(self.Mu0, Nu, Omega, self.Alpha0, self.Beta0, self.Kappa0, self.Ncell, self.Nsample)

    # Expectation of log P(Y|X,F)
    def Estep_PY(self, Nu, Omega, Beta):
        return Estep_PY_numba(self.Y, self.SigmaY, Nu, Omega, Beta, self.Ngene, self.Ncell, self.Nsample)

    # Expectation of log P(F)
    def Estep_PF(self, Beta):
        return -(np.sum(loggamma(self.Alpha)) - np.sum(loggamma(self.Alpha.sum(axis=1)))) + \
            np.sum((self.Alpha-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell])))

    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample*np.sum(np.log(Omega))

    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        return -(np.sum(loggamma(Beta)) - np.sum(loggamma(Beta.sum(axis=1))))+ \
            np.sum((Beta-1) * (digamma(Beta) - \
                np.tile(digamma(np.sum(Beta, axis=1))[:,np.newaxis], [1,self.Ncell]))
                )

    
    def grad_Nu(self, Nu, Omega, Beta): 
        # return Ngene by Ncell
        return grad_Nu_numba(self.Y, self.SigmaY, Nu, Omega, Beta, self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_numba(self.Y, self.SigmaY, Nu, Omega, Beta,
                          self.Mu0, self.Alpha0, self.Beta0, self.Kappa0, self.Ngene, self.Ncell, self.Nsample, self.weight)

    def g_Exp_Beta(self, Nu, Beta, B0):
        return g_Exp_Beta_numba(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

    def grad_Beta(self, Nu, Omega, Beta):
        # return Nsample by Ncell
        B0 = np.sum(self.Beta, axis=1)
        
        grad_PY = g_PY_Beta(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)

        grad_PF = (self.Alpha-1)*polygamma(1,Beta) - \
            np.tile(np.sum((self.Alpha-1)*np.tile(polygamma(1,B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        grad_QF = (Beta-1)*polygamma(1, Beta) - \
            np.tile(np.sum((Beta - 1) * np.tile(polygamma(1, B0)[:,np.newaxis], [1,self.Ncell]), axis=1)[:,np.newaxis], [1,self.Ncell])

        return grad_PY + grad_PF * np.sqrt(self.Ngene / self.Ncell) - grad_QF * np.sqrt(self.Ngene / self.Ncell)


    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1/self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        QX = self.Estep_QX(Omega) * (1/self.weight)
        QF = self.Estep_QF(Beta) * np.sqrt(self.Ngene / self.Ncell)

        return PX+PY+PF-QX-QF     

    def Optimize(self):
            
            # loss function
        def loss(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

            if self.Fix_par['Nu']:
                Nu = self.Nu
            if self.Fix_par['Beta']:
                Beta = self.Beta
            if self.Fix_par['Omega']:
                Omega = self.Omega

            return -self.E_step(Nu, Beta, Omega)

        # gradient function
        def grad(params):
            Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
            Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                    self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)
            
            if self.Fix_par['Nu']:
                g_Nu = np.zeros(Nu.shape)
            else:
                g_Nu = -self.grad_Nu(Nu, Omega, Beta)
            
            if self.Fix_par['Omega']:
                g_Omega = np.zeros(Omega.shape)
            else:
                g_Omega = -self.grad_Omega(Nu, Omega, Beta)
            
            if self.Fix_par['Beta']:
                g_Beta = np.zeros(Beta.shape)
            else:
                g_Beta = -self.grad_Beta(Nu, Omega, Beta)

            g = np.concatenate((g_Nu.flatten(), g_Omega.flatten(), g_Beta.flatten()))

            return g

        # Perform Optimization
        Init = np.concatenate((self.Nu.flatten(), self.Omega.flatten(), self.Beta.flatten()))
        bounds = [(-np.inf, np.inf) if i < (self.Ncell*self.Ngene*self.Nsample) else (0.0000001, 100) for i in range(len(Init))]

        out = scipy.optimize.minimize(
                fun = loss, x0 = Init, bounds = bounds, jac = grad,
                options = {'disp': False},
                method='L-BFGS-B')

        params = out.x
        
        
        
        
        self.Nu = params[0:self.Ncell*self.Ngene*self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
        self.Omega = params[self.Ncell*self.Ngene*self.Nsample:(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell)].reshape(self.Ngene, self.Ncell)
        self.Beta = params[(self.Ncell*self.Ngene*self.Nsample + self.Ngene*self.Ncell):(self.Ncell*self.Ngene*self.Nsample + \
                        self.Ngene*self.Ncell + self.Nsample*self.Ncell)].reshape(self.Nsample, self.Ncell)

        self.log = out.success      
     
    # Reestimation of Nu at specific weight
    def Reestimate_Nu(self,weight=100):
        self.weight=weight
        self.Optimize()
        return self
   
    def Check_health(self):
        # check if optimization is done
        if not hasattr(self, 'log'):
            warnings.warn("No optimization is not done yet", Warning, stacklevel=2)

        # check values in hyperparameters
        if not np.all(np.isfinite(self.Y)):
            warnings.warn('non-finite values detected in bulk gene expression data (Y).', Warning, stacklevel=2)
            
        if np.any(self.Y < 0):
            warnings.warn('Negative expression levels were detected in bulk gene expression data (Y).', Warning, stacklevel=2)

        if np.any(self.Alpha <= 0):
            warnings.warn('Zero or negative values in Alpha', Warning, stacklevel=2)

        if np.any(self.Beta <= 0):
            warnings.warn('Zero or negative values in Beta', Warning, stacklevel=2)
 
        if np.any(self.Alpha0 <= 0):
            warnings.warn('Zero or negative values in Alpha0', Warning, stacklevel=2)
        
        if np.any(self.Beta0 <= 0):
            warnings.warn('Zero or negative values in Beta0', Warning, stacklevel=2)
        
        if np.any(self.Kappa0 <= 0):
            warnings.warn('Zero or negative values in Kappa0', Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta)
        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = np.identity(Expected['Expectation'].shape[1])
                Expected = Expected['Expectation']
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                IndG = np.where(~np.isnan(Expected[sample,:]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group,:] == 1)[0]
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] / np.sum(Fraction[sample,IndCell])  # make fraction sum to one for the group
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] * Expected[sample, group]  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[sample, IndNan] = Fraction[sample, IndNan] / np.sum(Fraction[sample, IndNan])  # normalize the rest of cell types (sum to one)
                Fraction[sample, IndNan] = Fraction[sample, IndNan] * (1-np.sum(Expected[sample, IndG]))  # assign determined fraction for the rest of cell types

        if Temperature is not None:
            self.Alpha = Temperature * Fraction
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = Fraction[sample,:] * np.sum(self.Beta[sample,:])

    def Update_Alpha_Group(self, Expected=None, Temperature=None):# if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = np.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / np.sum(AvgBeta)

        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if "Group" in Expected:  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected['Group']
                else:
                    Group = np.identity(Expected['Expectation'].shape[1])
                Expected = Expected['Expectation']
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError('Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)')

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = np.copy(Fraction_Avg)
                IndG = np.where(~np.isnan(Expected[sample,:]))[0]
                IndCells = []
                
                for group in IndG:
                    IndCell = np.where(Group[group,:] == 1)[0]
                    Fraction[IndCell] = Fraction[IndCell] / np.sum(Fraction[IndCell])  # make fraction sum to one for the group
                    Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)
                    
                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[IndNan] = Fraction[IndNan] / np.sum(Fraction[IndNan])  # normalize the rest of cell types (sum to one)
                Fraction[IndNan] = Fraction[IndNan] * (1-np.sum(Expected[sample, IndG]))  # assign determined fraction for the rest of cell types
            
                AlphaSum = np.sum(AvgBeta[IndNan])/ np.sum(Fraction[IndNan])
                self.Alpha[sample, :] = Fraction * AlphaSum
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample,:] = AvgBeta

    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        
        a = Var / Exp / Exp
        b = np.square((self.Y-np.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = np.sqrt(a+b)
        else:  # shared in all samples
            self.SigmaY = np.tile(np.mean(np.sqrt(a+b), axis=1)[:,np.newaxis], [1,self.Nsample])




def Parallel_Purification(obj,weight, iter=1000, minDiff=10e-4, Update_SigmaY=False):
    obj.Check_health()
    obj_func = [float('nan')] * iter
    obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
    for i in range(1, iter):
        obj.Reestimate_Nu(weight=weight)
        if Update_SigmaY:
            obj.Update_SigmaY()
        obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        # Check for convergence
        if np.abs(obj_func[i] - obj_func[i-1]) < minDiff:
            break
    return obj, obj_func

def Purify_AllGenes(BLADE_object, Mu, Omega, Y, Ncores,Weight=100,sY = 1,Alpha0 = 1000,Kappa0 = 1):
    Mu = ensure_numpy(Mu)
    Omega = ensure_numpy(Omega)
    Y = ensure_numpy(Y)
    obj = BLADE_object
    obj.Alpha = convert_to_numpy(obj.Alpha)
    obj.SigmaY = convert_to_numpy(obj.SigmaY)
    obj.Mu0 = convert_to_numpy(obj.Mu0)
    obj.Beta0 = convert_to_numpy(obj.Beta0)
    obj.Kappa0 = convert_to_numpy(obj.Kappa0)
    obj.Beta = convert_to_numpy(obj.Beta)
    
    Ngene, Nsample = Y.shape
    Ncell = Mu.shape[1]
    logY = np.log(Y+1)
    SigmaY = np.tile(np.std(logY,1)[:,np.newaxis], [1,Nsample]) * sY + 0.1
    Beta0 = Alpha0 * np.square(Omega)
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i,:,:] = Mu

    # Fetch objs per gene
    Ngene_total = Mu.shape[0]
    objs = []
    for ix in range(Ngene_total):
        objs.append(BLADE_numba(
            Y = np.atleast_2d(logY[ix,:]),
            SigmaY = np.atleast_2d(SigmaY[ix,:]),
            Mu0 = np.atleast_2d(Mu[ix,:]),
            Alpha = obj.Alpha,
            Alpha0 = Alpha0,
            Beta0 = np.atleast_2d(Beta0[ix,:]),
            Kappa0 =Kappa0,
            Nu_Init = np.reshape(np.atleast_3d(Nu_Init[:,ix,:]), (Nsample,1,Ncell)), 
            Omega_Init = np.atleast_2d(Omega[ix,:]),
            Beta_Init = obj.Beta,
            fix_Beta=True))

    outs = Parallel(n_jobs=Ncores, verbose=10)(
                delayed(Parallel_Purification)(obj,Weight)
                    for obj in objs
                )
       
    objs, obj_func = zip(*outs)
    ## sum ofv over all genes
    obj_func = np.sum(obj_func, axis = 0)
    logs = []
    ## Combine results from all genes
    for i,obj in enumerate(objs):
        logs.append(obj.log)
        if i==0:
            Y = objs[0].Y
            SigmaY = objs[0].SigmaY
            Mu0= objs[0].Mu0
            Alpha = objs[0].Alpha
            Alpha0 = objs[0].Alpha0
            Beta0 = objs[0].Beta0
            Kappa0 = objs[0].Kappa0
            Nu_Init = objs[0].Nu
            Omega_Init = objs[0].Omega
            Beta_Init = objs[0].Beta
        else:    
            Y = np.concatenate((Y,obj.Y))
            SigmaY = np.concatenate((SigmaY,obj.SigmaY))
            Mu0= np.concatenate((Mu0,obj.Mu0))
            Alpha0 = np.concatenate((Alpha0,obj.Alpha0))
            Beta0 = np.concatenate((Beta0,obj.Beta0))
            Kappa0 = np.concatenate((Kappa0,obj.Kappa0))
            Nu_Init = np.concatenate((Nu_Init,obj.Nu), axis = 1)
            Omega_Init = np.concatenate((Omega_Init,obj.Omega))
            
    ## Create final merged BLADE obj to return
    obj = BLADE_numba(Y, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, fix_Beta =True)
    obj.log = logs
    
    return obj









