import os

# Disable CUDA graphs in Inductor — safer under multi-thread/multi-proc joblib
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

TMP = os.environ.get("JOBLIB_TEMP_FOLDER", "/dev/shm")

# New imports
import torch
import torch.special

import dill

# Old imports
from numba import jit, njit

###NumPy import
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

import importlib
from contextlib import contextmanager


if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
else:
    torch.set_default_dtype(torch.float32)


def _debug_precision_report(model, tag="[Precision]"):
    print(f"{tag} Backend: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"{tag} Default dtype: {torch.get_default_dtype()}")
    for name in ("Nu", "Omega", "Beta"):
        p = getattr(model, name, None)
        if isinstance(p, torch.Tensor):
            print(f"{tag} Param {name}: dtype={p.dtype}, device={p.device}")


# --------- re-entrant-safe, CPU-friendly profiler (replace previous block) ---------
import os, io, time, functools, contextvars, torch
from torch.profiler import profile, ProfilerActivity

# === config ===
# defaults for profiling cpu/GPU time and memory usage
PROF_ENABLED = False
PROF_SAVE_TXT = ""  # e.g. "profiles.txt" to append; "" → print only
PROF_ROW_LIMIT = 40
PROF_RECORD_SHAPES = True
PROF_PROFILE_MEM = True
PROF_WITH_STACK = False
PROF_SORT = None  # None→ auto: cuda_time_total if CUDA else self_cpu_time_total
PROF_ONLY_FIRST_N = 1  # profile only the first N calls per function per process
PROF_SKIP_NESTED = True  # <<< IMPORTANT: avoid nested Kineto sessions
PROF_ALLOW_CUDA = True  # auto-add CUDA activity if available

###=== example config for CPU-only, no file output ===
PROF_ALLOW_CUDA = False
PROF_SORT = ""
PROF_SAVE_TXT = ""
PROF_ONLY_FIRST_N = 1

"""
Use as a function decorator, just add @torch_prof on top of the function definition. It will automatically profile the function using torch.profiler and print/save the results.
"""

# process-local nesting depth (survives joblib workers)
_prof_depth = contextvars.ContextVar("prof_depth", default=0)


def _sort_default():
    return "cuda_time_total" if (torch.cuda.is_available() and PROF_ALLOW_CUDA) else "self_cpu_time_total"


def _emit(fn_name: str, kind: str, text: str):
    header = f"\n[{kind}] {fn_name} @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if PROF_SAVE_TXT:
        with open(PROF_SAVE_TXT, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    else:
        print(header + text)


def torch_prof(fn):
    calls = {"n": 0}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if (not PROF_ENABLED) or (calls["n"] >= PROF_ONLY_FIRST_N):
            return fn(*args, **kwargs)

        depth = _prof_depth.get()
        if PROF_SKIP_NESTED and depth > 0:
            # Already inside a profiler in this process → just run
            return fn(*args, **kwargs)

        calls["n"] += 1
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and PROF_ALLOW_CUDA:
            activities.append(ProfilerActivity.CUDA)

        _prof_depth.set(depth + 1)
        try:
            # Run under profiler; if Kineto misbehaves, we’ll fall back gracefully
            with profile(
                activities=activities,
                record_shapes=PROF_RECORD_SHAPES,
                profile_memory=PROF_PROFILE_MEM,
                with_stack=PROF_WITH_STACK,
            ) as prof:
                out = fn(*args, **kwargs)
        except RuntimeError as e:
            # Handle Kineto hiccups (esp. on CPU-only or exotic builds)
            if "Kineto" in str(e):
                _emit(fn.__name__, "torch.profiler (skipped)", f"Skipped due to Kineto error: {e}")
                return fn(*args, **kwargs)
            raise
        finally:
            # make sure we always unwind the depth
            _prof_depth.set(depth)

        sort_by = PROF_SORT or _sort_default()
        table = prof.key_averages().table(sort_by=sort_by, row_limit=PROF_ROW_LIMIT)
        _emit(fn.__name__, "torch.profiler", table)
        return out

    return wrapped


# Optional: Python-level profiler (safe to keep as-is)
def py_prof(fn):
    import cProfile, pstats

    calls = {"n": 0}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if (not PROF_ENABLED) or (calls["n"] >= PROF_ONLY_FIRST_N):
            return fn(*args, **kwargs)
        calls["n"] += 1
        pr = cProfile.Profile()
        pr.enable()
        try:
            return fn(*args, **kwargs)
        finally:
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(PROF_ROW_LIMIT)
            _emit(fn.__name__, "cProfile", s.getvalue())

    return wrapped


# -------------------------------------------------------------------------------

# guards
EXP_MAX = 80.0  # cap exponent arguments (float32-safe)
EPS = 1e-12  # safe minimum for divides/log/exp

"""
Use as a function decorator, just add @numerical_guard on top of the function definition. It will automatically clamp the input of torch.exp, torch.log and denominator of torch.div/true_divide to avoid overflow/NaN.
Adds 35% overhead in our tests, so use only around numerically sensitive code.         
"""


class numerical_guard:
    """
    Context manager for guarding torch.exp / torch.log / division.
    Usage:
        with numerical_guard():
            out = my_fn(...)
    """

    def __enter__(self):
        self._old = {
            "exp": torch.exp,
            "log": torch.log,
            "div": torch.div,
            "true_divide": torch.true_divide,
            "Tensor_log_": getattr(torch.Tensor, "log_", None),
            "Tensor_exp_": getattr(torch.Tensor, "exp_", None),
        }

        old_exp = self._old["exp"]
        old_log = self._old["log"]
        old_div = self._old["div"]
        old_true_div = self._old["true_divide"]

        def _safe_exp(x):
            try:
                if x.is_floating_point():
                    x = x.clamp_max(EXP_MAX)
            except AttributeError:
                pass
            return old_exp(x)

        def _safe_log(x):
            try:
                if x.is_floating_point():
                    x = x.clamp_min(EPS)
            except AttributeError:
                pass
            return old_log(x)

        def _guard_den(b):
            # preserve sign and clamp magnitude away from 0
            if torch.is_tensor(b):
                if b.is_floating_point():
                    return torch.where(b >= 0, b.clamp_min(EPS), b.clamp_max(-EPS))
                return b
            # python scalar
            if isinstance(b, (float, int)):
                b = float(b)
                if b >= 0.0:
                    return b if b >= EPS else EPS
                else:
                    return b if b <= -EPS else -EPS
            return b

        def _safe_div(a, b):
            return old_true_div(a, _guard_den(b))

        torch.exp = _safe_exp
        torch.log = _safe_log
        torch.div = _safe_div
        torch.true_divide = _safe_div

        # best-effort: patch in-place tensor methods (calls still go through aten, so this is a bonus)
        if self._old["Tensor_log_"] is not None:

            def _tensor_log_(t):
                if t.is_floating_point():
                    t.clamp_min_(EPS)
                return self._old["Tensor_log_"](t)

            torch.Tensor.log_ = _tensor_log_

        if self._old["Tensor_exp_"] is not None:

            def _tensor_exp_(t):
                if t.is_floating_point():
                    t.clamp_max_(EXP_MAX)
                return self._old["Tensor_exp_"](t)

            torch.Tensor.exp_ = _tensor_exp_

        return self

    def __exit__(self, exc_type, exc, tb):
        torch.exp = self._old["exp"]
        torch.log = self._old["log"]
        torch.div = self._old["div"]
        torch.true_divide = self._old["true_divide"]
        if self._old["Tensor_log_"] is not None:
            torch.Tensor.log_ = self._old["Tensor_log_"]
        if self._old["Tensor_exp_"] is not None:
            torch.Tensor.exp_ = self._old["Tensor_exp_"]


def guard_numeric(fn):
    """
    Decorator version of the numerical guard.
    Usage:
        @guard_numeric
        def my_fn(...): ...
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with numerical_guard():
            return fn(*args, **kwargs)

    return wrapped


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


# @torch_prof
def VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # --- shape checks ---
    S, G, C = Nu.shape
    assert (S, G, C) == (Nsample, Ngene, Ncell), "Shape mismatch vs. Ngene/Ncell/Nsample"

    EPS = 1e-12
    device, dtype = Nu.device, Nu.dtype

    # ---- Dirichlet bits (computed once) ----
    B0 = Beta.sum(dim=1, keepdim=True).clamp_min(EPS)  # (S,1)
    mu = ExpF_C(Beta)  # (S,C)
    mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(EPS)
    mu2 = mu * mu  # (S,C)
    VarB = mu * (1.0 - mu) / (B0 + 1.0)  # (S,C)

    # ---- helper: (S,C) x (S,g,C) -> (S,g) via bmm ----
    def _bmm_sum_over_c(weights_sc, mat_sgc):
        return torch.bmm(
            weights_sc.unsqueeze(1),  # (S,1,C)
            mat_sgc.transpose(1, 2).contiguous(),  # (S,C,g)
        ).squeeze(1)  # (S,g)

    # ---- output buffer ----
    out = torch.empty((G, S), dtype=dtype, device=device)

    # ---- choose chunk size for genes to keep ~64 MB live working set ----
    # live big tensors per chunk: v (S,g,C) and v2 (S,g,C) ⇒ about 2 * S*g*C * element_size bytes
    elt = Nu.element_size()
    cap_bytes = 64 * 1024 * 1024
    denom = max(1, 3 * S * C * elt)  # use 3× for safety (accounts for small extras)
    chunk_G = max(1, cap_bytes // denom)

    one_over_1pB0 = 1.0 / (1.0 + B0)  # (S,1)

    for g0 in range(0, G, chunk_G):
        g1 = min(G, g0 + chunk_G)
        Nu_s = Nu[:, g0:g1, :]  # (S,g,C)
        Om_s = Omega[g0:g1, :]  # (g,C)
        Om2 = Om_s.square()  # (g,C)

        # v  = E[X]   = exp(Nu + 0.5*Omega^2)
        # v2 = (E[X])^2 = exp(2*Nu + Omega^2)
        v = torch.exp(Nu_s + 0.5 * Om2.unsqueeze(0))  # (S,g,C)
        v2 = v.square()  # (S,g,C)

        # E[X^2] = exp(2*Nu + 2*Omega^2) = v2 * exp(Omega^2)
        expX2 = v2 * torch.exp(Om2).unsqueeze(0)  # (S,g,C)

        # ---- VarTerm ----
        # T1 = sum_c E[X^2] * (VarF + mu^2)
        T1 = _bmm_sum_over_c(VarB + mu2, expX2)  # (S,g)
        # T2 = sum_c (E[X])^2 * mu^2
        T2 = _bmm_sum_over_c(mu2, v2)  # (S,g)
        VarTerm_Sg = T1 - T2  # (S,g)

        # ---- CovTerm (closed-form; no 4-D, no loops) ----
        # sum_mu_v  = sum_c mu * E[X]
        # sum_mu2v2 = sum_c mu^2 * (E[X])^2
        sum_mu_v = _bmm_sum_over_c(mu, v)  # (S,g)
        sum_mu2v2 = _bmm_sum_over_c(mu2, v2)  # (S,g)
        Cov_Sg = -(sum_mu_v.square() - sum_mu2v2) * one_over_1pB0  # (S,g) / (1+B0)

        # write chunk as (G_chunk, S)
        out[g0:g1, :] = (VarTerm_Sg + Cov_Sg).T

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

    # ---------- 1) grad_PX (same math as your original) ----------
    AlphaN = Alpha0 + S * 0.5
    NuExp = torch.sum(Nu, dim=0) / S  # (G,C)

    # Accumulator for ExpBetaN (exact same formula, vectorized)
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

    # ---------- 5) g_Exp wrt Nu (same as your code, just vectorized) ----------
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

    # ---------- 7) Assemble a, b, grad_PY (same formulas as your code) ----------
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

    # -------- 4) Expectation/variance of Q (as in your pipeline) --------
    Exp = ExpQ_C(Nu, Beta, Omega)  # (G,S)
    Var = VarQ_C(Nu, Beta, Omega, G, C, S)  # (G,S)

    # -------- 5) g_Exp term (same as your code, but vectorized) --------
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

    # -------- 9) Q(X) term (same as original) --------
    grad_QX = -S / Omega  # (G,C)

    # -------- 10) Final combine (same scaling by weight) --------
    return grad_PX * (1.0 / weight) + grad_PY - grad_QX * (1.0 / weight)


# @torch_prof
def g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    # Compute exp(Nu) element-wise
    ExpX = torch.exp(Nu)  # Shape: (Nsample, Ngene, Ncell)
    # Apply the element-wise multiplication with Omega, which has shape (Ngene, Ncell)
    ExpX = ExpX * torch.exp(0.5 * torch.square(Omega)).unsqueeze(0)

    # B0mat computation (element-wise division)
    B0mat = Beta / torch.square(B0.unsqueeze(1))  # Shape: (Nsample, Ncell)

    # Transpose ExpX to match the original NumPy (Nsample, Ncell, Ngene)
    tExpX = ExpX.transpose(1, 2)  # Shape: (Nsample, Ncell, Ngene)

    # Perform dot product of B0mat and tExpX
    B0mat = torch.matmul(B0mat.unsqueeze(1), tExpX).squeeze(1)  # Shape: (Nsample, 1, Ngene)

    # Divide ExpX by the broadcasted B0 (shape should match (Nsample, Ngene, Ncell))
    g_Exp = ExpX / B0.unsqueeze(1).unsqueeze(2)  # Shape: (Nsample, Ngene, Ncell)

    # Subtract B0mat which has shape (Nsample, Ngene), broadcast it over Ncell
    g_Exp = g_Exp - B0mat.unsqueeze(2)  # Shape: (Nsample, Ngene, Ncell)
    return g_Exp.permute(0, 2, 1)


# @torch_prof
def g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    """
    Drop-in replacement using batched GEMMs over the cell axis (C).
    Shapes:
      Nu:    (Nsample, Ngene, Ncell)
      Omega: (Ngene,  Ncell)
      Beta:  (Nsample, Ncell)
      B0:    (Nsample,)
    Returns:
      g_Var: (Nsample, Ncell, Ngene)   # same as original
    """
    S, G, C = Nsample, Ngene, Ncell
    assert Nu.shape == (S, G, C) and Omega.shape == (G, C) and Beta.shape == (S, C)
    assert B0.shape == (S,)

    # helpers
    def _bmm_sum_over_c(W_sc: torch.Tensor, M_sgc: torch.Tensor) -> torch.Tensor:
        # (S,C) x (S,G,C) -> (S,G)
        return torch.bmm(W_sc.unsqueeze(1), M_sgc.transpose(1, 2).contiguous()).squeeze(1)

    B0Rep = B0.unsqueeze(1)  # (S,1)
    B0B0_1 = B0Rep * (B0Rep + 1.0)  # (S,1)

    # ----- aa and aaNotT (exact formulas from original) -----
    aa = (B0Rep - Beta) * B0Rep * (B0Rep + 1.0) - (3.0 * B0Rep + 2.0) * Beta * (B0Rep - Beta)
    aa = aa / (B0Rep.pow(3) * (B0Rep + 1.0).square())
    aa = aa + 2.0 * Beta * (B0Rep - Beta) / B0Rep.pow(3)

    aaNotT = Beta * B0Rep * (B0Rep + 1.0) - (3.0 * B0Rep + 2.0) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (B0Rep.pow(3) * (B0Rep + 1.0).square())
    aaNotT = aaNotT + 2.0 * Beta * (-Beta) / B0Rep.pow(3)

    # ----- Exp terms -----
    ExpX2 = torch.exp(2.0 * Nu + 2.0 * Omega.unsqueeze(0).square())  # (S,G,C)
    # base term: transpose to (S,C,G) then scale by aa (S,C,1)
    g_Var = ExpX2.transpose(1, 2) * aa.unsqueeze(2)  # (S,C,G)

    # add aaNotT contributions excluding the diagonal:
    # sum_j ExpX2[:,:,j]*aaNotT[:,j]   minus   ExpX2[:,:,i]*aaNotT[:,i]
    sum_all = _bmm_sum_over_c(aaNotT, ExpX2)  # (S,G)
    diag_nt = (aaNotT.unsqueeze(1) * ExpX2).transpose(1, 2)  # (S,C,G)
    g_Var = g_Var + sum_all.unsqueeze(1) - diag_nt  # (S,C,G)

    # ----- Element-wise Beta terms -----
    B_B02 = Beta / B0Rep.square()  # (S,C)
    B2_B03 = Beta.square() / B0Rep.pow(3)  # (S,C)

    ExpX_2nu_om2 = torch.exp(2.0 * Nu + Omega.unsqueeze(0).square())  # (S,G,C)

    # subtract 2 * ExpX term (diagonal per cell)
    g_Var = g_Var - 2.0 * (ExpX_2nu_om2.transpose(1, 2) * B_B02.unsqueeze(2))  # (S,C,G)

    # add 2 * dot term over c
    Dot = _bmm_sum_over_c(B2_B03, ExpX_2nu_om2)  # (S,G)
    g_Var = g_Var + 2.0 * Dot.unsqueeze(1)  # (S,C,G)

    # ----- Covariance-type terms without 4-D -----
    v = torch.exp(Nu + 0.5 * Omega.unsqueeze(0).square())  # (S,G,C)

    # CovTerm1: sum_{l!=k} gradCovB_{l,k} * v_l * v_k
    # gradCovB_{l,k} = Beta_l * Beta_k * ((3B0+2)/(B0^3*(B0+1)^2))
    c1 = (3.0 * B0 + 2.0) / (B0.pow(3) * (B0 + 1.0).square())  # (S,)
    sum_Bv = _bmm_sum_over_c(Beta, v)  # (S,G)
    sum_B2v2 = _bmm_sum_over_c(Beta.square(), v.square())  # (S,G)
    CovTerm1_SG = c1.unsqueeze(1) * (sum_Bv.square() - sum_B2v2)  # (S,G)
    g_Var = g_Var + CovTerm1_SG.unsqueeze(1)  # (S,C,G)  (broadcast over C)

    # CovTerm2: for each k, sum_{l!=k} B_w[l] * v_l * v_k
    # where B_w = Beta * (B0+1) / (B0*(B0+1))^2 = Beta*(B0+1) / (B0B0_1^2)
    B_w = Beta * (B0Rep + 1.0) / B0B0_1.square()  # (S,C)
    sum_Bw_v = _bmm_sum_over_c(B_w, v)  # (S,G)
    CovTerm2_SGC = v * sum_Bw_v.unsqueeze(2) - (B_w.unsqueeze(1) * v.square())  # (S,G,C)
    g_Var = g_Var - 2.0 * CovTerm2_SGC.permute(0, 2, 1)  # (S,C,G)

    return g_Var


# @torch_prof
def g_PY_Beta_C(Nu, Beta, Omega, Y, SigmaY, B0, Ngene, Ncell, Nsample):
    # 1) Compute Exp, Var (shapes: (Ngene, Nsample))
    Exp = ExpQ_C(Nu, Beta, Omega)
    Var = VarQ_C(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    # 2) Compute g_Exp, g_Var (shapes: (Nsample, Ncell, Ngene))
    g_Exp = g_Exp_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)
    g_Var = g_Var_Beta_C(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample)

    # 3) "a" term
    #    a[s,c,g] = (g_Var[s,c,g]*Exp[g,s] - 2*g_Exp[s,c,g]*Var[g,s]) / Exp[g,s]^3
    #
    # We reorder Exp->(Nsample,Ngene) so it can broadcast with (Nsample,Ncell,Ngene).
    Exp_t = Exp.permute(1, 0)  # (Nsample, Ngene)
    Var_t = Var.permute(1, 0)  # (Nsample, Ngene)

    numerator = g_Var * Exp_t.unsqueeze(1) - 2.0 * g_Exp * Var_t.unsqueeze(1)
    denominator = Exp_t.unsqueeze(1).pow(3)  # Exp[g,s]^3
    a = numerator / denominator  # (Nsample, Ncell, Ngene)

    # 4) "b" term
    #    varExp2[g,s] = Var[g,s]/(2 * Exp[g,s]^2)
    #    b[s,c,g]     = - (Y[g,s] - log(Exp[g,s]) - varExp2[g,s])
    #                     * [2*g_Exp[s,c,g]/Exp[g,s] + a[s,c,g]]
    #
    varExp2 = Var / (2.0 * Exp.square())  # (Ngene, Nsample)
    varExp2_t = varExp2.permute(1, 0)  # (Nsample, Ngene)

    Y_t = Y.permute(1, 0)  # (Nsample, Ngene)
    logExp_t = torch.log(Exp).permute(1, 0)  # (Nsample, Ngene)

    two_gExp_over_Exp = 2.0 * g_Exp / Exp_t.unsqueeze(1)
    inside = two_gExp_over_Exp + a

    diff = Y_t.unsqueeze(1) - logExp_t.unsqueeze(1) - varExp2_t.unsqueeze(1)
    b = -diff * inside  # (Nsample, Ncell, Ngene)

    # 5) Combine (a + b), multiply by (0.5 / SigmaY^2), sum over gene
    sum_ab = a + b
    SigmaY_sq = SigmaY.square()  # (Ngene, Nsample)
    SigmaY_sq_t = SigmaY_sq.permute(1, 0).unsqueeze(1)  # (Nsample, 1, Ngene)

    factor = 0.5 / SigmaY_sq_t
    weighted_sum_ab = factor * sum_ab  # (Nsample, Ncell, Ngene)
    grad_PY = -torch.sum(weighted_sum_ab, dim=2)  # sum over gene => (Nsample, Ncell)

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

    def push(self, *, outer_step: int, phase: str, obj_val: float, grad_norms: dict | None = None, note: str = ""):
        if self._t0 is None:
            self.start()
        grad_norms = grad_norms or {}
        self.records.append(
            {
                "t": time.perf_counter() - self._t0,  # seconds since first push
                "step": int(outer_step),
                "phase": str(phase),
                "obj": float(obj_val),  # ELBO
                "gNu": float(grad_norms.get("Nu", 0.0)),
                "gOm": float(grad_norms.get("Omega", 0.0)),
                "gBe": float(grad_norms.get("Beta", 0.0)),
                "closures": int(self._closure_calls),
                "note": note,
            }
        )


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
        import torch

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )

        # 👇 fix: declare global inside the function

        # 2) Keep a weight tensor (we'll convert dtype later in one go)
        self.weight = torch.tensor(1, device=self.device)

        # 3) Core tensors (initially as torch tensors; dtype normalized later)
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

        # # Debug precision report (only print once per process)
        # if not hasattr(BLADE, "_precision_reported"):
        #     _debug_precision_report(self)
        #     BLADE._precision_reported = True

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
        grad_PY = g_PY_Beta_C(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)
        # print(grad_PY, "grad_PY")
        # 3. Compute grad_PF
        polygamma_Beta = torch.special.polygamma(1, Beta)  # (Nsample, Ncell)
        polygamma_B0 = torch.special.polygamma(1, B0).unsqueeze(1)  # (Nsample, 1)

        grad_PF = (self.Alpha - 1) * polygamma_Beta - torch.sum((self.Alpha - 1) * polygamma_B0, dim=1, keepdim=True)

        # print(grad_PF, "grad_PF")

        # 4. Compute grad_QF
        grad_QF = (Beta - 1) * polygamma_Beta - torch.sum((Beta - 1) * polygamma_B0, dim=1, keepdim=True)

        # print(grad_QF, "grad_QF")

        # 5. Combine everything (same final scaling as in NumPy code)
        scaling_factor = torch.sqrt(torch.tensor(self.Ngene / self.Ncell, dtype=Beta.dtype, device=Beta.device))

        # print(grad_PY + grad_PF * scaling_factor - grad_QF * scaling_factor, "grad_Beta")

        return grad_PY + grad_PF * scaling_factor - grad_QF * scaling_factor

    # @torch_prof
    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1 / self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        QX = self.Estep_QX(Omega) * (1 / self.weight)
        QF = self.Estep_QF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        return PX + PY + PF - QX - QF

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
            if isinstance(self.Beta, torch.Tensor):
                self.Beta.clamp_(min=1e-7, max=100.0)

            # 3) Do NOT clamp Nu (unbounded in the old SciPy version)
            #    (We rely on internal exp-guards like EXP_MAX inside math kernels.)

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
        phase = opt_kwargs.pop("phase", None) or method.lower()
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
                return torch.amp.autocast("cuda", dtype=dtype)
            except AttributeError:
                return torch.cuda.amp.autocast(dtype=dtype)

        amp_grads = bool(opt_kwargs.pop("amp_grads", True))
        amp_ctx = _amp_autocast_grad(enabled=amp_grads, dtype=torch.bfloat16)

        # --- 0) make sure starting point is sane (once) ---
        self._finite_clamp_()

        # --- 1) prepare trainables respecting Fix_par ---
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

        # --- 2) choose optimizer (thread through **opt_kwargs) ---
        method_l = method.lower()
        lbfgs_allowed = {
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
        adam_allowed = {
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "capturable",
            "foreach",
            "maximize",
            "differentiable",
            "fused",
            "lr",
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
            if not self.Fix_par["Nu"] and getattr(self.Nu, "grad", None) is not None:
                d["Nu"] = self.Nu.grad.norm().item()
            if not self.Fix_par["Omega"] and getattr(self.Omega, "grad", None) is not None:
                d["Omega"] = self.Omega.grad.norm().item()
            if not self.Fix_par["Beta"] and getattr(self.Beta, "grad", None) is not None:
                d["Beta"] = self.Beta.grad.norm().item()
            return d

        # --- 3) LBFGS closure (analytical grads; E_step FP32; no clamp here) ---
        last_grad_norms = {}

        def closure():
            if logger:
                logger.bump_closure()
            opt.zero_grad(set_to_none=True)

            obj = self.E_step(self.Nu, self.Beta, self.Omega)
            if not torch.isfinite(obj):
                self._finite_clamp_()
                # fallback in correct dtype
                return torch.tensor(1e30, device=self.device, dtype=self.Nu.dtype)

            with amp_ctx:
                self._analytical_grads_()
            _clean_grads(do_clip=False)

            nonlocal last_grad_norms
            last_grad_norms = _grad_norms_dict()
            return -obj  # LBFGS minimizes

        # --- 4) main loop ---
        if logger:
            logger.start()
        for _ in range(steps):
            if use_lbfgs:
                with torch.no_grad():
                    snapshot = [p.clone() for p in trainable]
                try:
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
                    grad_norms=last_grad_norms,
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
    def Reestimate_Nu(self, weight=100):
        self.weight = weight
        self.Optimize()
        return self

    def Check_health(self):
        # check if optimization is done
        if not hasattr(self, "log"):
            warnings.warn("No optimization is not done yet", Warning, stacklevel=2)

        # check values in hyperparameters
        if not np.all(np.isfinite(self.Y.cpu().numpy())):
            warnings.warn("non-finite values detected in bulk gene expression data (Y).", Warning, stacklevel=2)

        if np.any(self.Y.cpu().numpy() < 0):
            warnings.warn(
                "Negative expression levels were detected in bulk gene expression data (Y).", Warning, stacklevel=2
            )

        if np.any(self.Alpha.cpu().numpy() <= 0):
            warnings.warn("Zero or negative values in Alpha", Warning, stacklevel=2)

        if np.any(self.Beta.cpu().numpy() <= 0):
            warnings.warn("Zero or negative values in Beta", Warning, stacklevel=2)

        if np.any(self.Alpha0.cpu().numpy() <= 0):
            warnings.warn("Zero or negative values in Alpha0", Warning, stacklevel=2)

        if np.any(self.Beta0.cpu().numpy() <= 0):
            warnings.warn("Zero or negative values in Beta0", Warning, stacklevel=2)

        if np.any(self.Kappa0.cpu().numpy() <= 0):
            warnings.warn("Zero or negative values in Kappa0", Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):  # if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta).cpu.numpy()
        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if (
                    "Group" in Expected
                ):  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected["Group"]
                else:
                    Group = np.identity(Expected["Expectation"].shape[1])
                Expected = Expected["Expectation"]
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError("Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)")

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                IndG = np.where(~np.isnan(Expected[sample, :]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group, :] == 1)[0]
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] / np.sum(
                        Fraction[sample, IndCell]
                    )  # make fraction sum to one for the group
                    Fraction[sample, IndCell] = (
                        Fraction[sample, IndCell] * Expected[sample, group]
                    )  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[sample, IndNan] = Fraction[sample, IndNan] / np.sum(
                    Fraction[sample, IndNan]
                )  # normalize the rest of cell types (sum to one)
                Fraction[sample, IndNan] = Fraction[sample, IndNan] * (
                    1 - np.sum(Expected[sample, IndG])
                )  # assign determined fraction for the rest of cell types

        if Temperature is not None:
            self.Alpha = Temperature * Fraction
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample, :] = Fraction[sample, :] * np.sum(self.Beta[sample, :])
        self.Alpha = torch.tensor(self.Alpha, device=self.device)

    def Update_Alpha_Group(self, Expected=None, Temperature=None):
        """
        Update Dirichlet prior α using group expectations.
        If Expected is None -> do nothing (keep current α).
        """
        if Expected is None:
            return  # keep self.Alpha as-is

        if isinstance(Expected, dict):
            Group = Expected.get("Group", torch.eye(Expected["Expectation"].shape[1], device=self.device))
            Expected = Expected["Expectation"]
        else:
            Group = torch.eye(Expected.shape[1], device=self.device)

        if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
            raise ValueError("Pre-determined fraction is in wrong shape (Nsample × Ncelltype)")

        Expected = torch.as_tensor(Expected, device=self.device, dtype=torch.float32)
        Group = torch.as_tensor(Group, device=self.device, dtype=torch.float32)

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

    def Update_Alpha_Group_old(
        self, Expected=None, Temperature=None
    ):  # if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = torch.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / torch.sum(AvgBeta)
        # print("Initial Fraction_Avg:", Fraction_Avg)

        if Expected is not None:  # Reflect the expected values
            # Expectation can be a dictionary (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if (
                    "Group" in Expected
                ):  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected["Group"]
                else:
                    Group = torch.eye(Expected["Expectation"].shape[1], device=self.device)
                Expected = Expected["Expectation"]
            else:
                Group = torch.eye(Expected.shape[1], device=self.device)

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError("Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)")
            Expected = torch.tensor(Expected, device=self.device)
            # print("Expected values:\n", Expected)
            # print("Group matrix:\n", Group)

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = Fraction_Avg.clone()
                # print(f"\nSample {sample}:")
                # print("  Initial Fraction:", Fraction)

                # Get indices of expected cell types (non-NaN)
                IndG = torch.where(~torch.isnan(Expected[sample, :]))[0]
                # print("  IndG (indices with expectations):", IndG)

                IndCells = []
                for group in IndG:
                    IndCell = torch.where(Group[group, :] == 1)[0]
                    # print("    Group:", group.item(), "-> IndCell:", IndCell)
                    # Normalize fractions in the group to sum to 1
                    Fraction[IndCell] = Fraction[IndCell] / torch.sum(Fraction[IndCell])
                    # print("    Normalized Fraction for group:", Fraction[IndCell])
                    # Multiply by the expected value for that group
                    Fraction[IndCell] = Fraction[IndCell] * Expected[sample, group]
                    # print("    Adjusted Fraction for group:", Fraction[IndCell])
                    IndCells.extend(IndCell.tolist())

                IndNan = torch.tensor(list(set(range(Group.shape[1])) - set(IndCells)), device=Fraction.device)
                # print("  IndNan (cell types with no expectation):", IndNan)
                Fraction[IndNan] = Fraction[IndNan] / torch.sum(Fraction[IndNan])
                Fraction[IndNan] = Fraction[IndNan] * (1 - torch.sum(Expected[sample, IndG]))
                # print("  Adjusted Fraction for non-specified cells:", Fraction[IndNan])

                AlphaSum = torch.sum(AvgBeta[IndNan]) / torch.sum(Fraction[IndNan])
                # print("  AlphaSum:", AlphaSum)
                self.Alpha[sample, :] = Fraction * AlphaSum
                # print("  Updated Alpha for sample:", self.Alpha[sample, :])
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample, :] = AvgBeta

    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_C(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_C(self.Nu, self.Beta, self.Omega)

        a = Var / Exp / Exp
        b = torch.square((self.Y - torch.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = torch.sqrt(a + b)
        else:  # shared in all samples
            self.SigmaY = torch.mean(torch.sqrt(a + b), dim=1, keepdim=True).expand(-1, self.Nsample)


def Optimize(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Nsample, Ncell, Init_Fraction):
    Beta_Init = np.random.gamma(shape=1, size=(Nsample, Ncell)) * 0.1 + t(Init_Fraction) * 10
    obs = BLADE(
        logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, fix_Nu=True, fix_Omega=True
    )
    obs.Optimize()
    obs.Fix_par["Nu"] = False
    obs.Fix_par["Omega"] = False
    obs.Optimize()
    return obs


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
    iter=100,
    minDiff=1e-4,
    TempRange=None,
    Update_SigmaY=False,
    device=None,
    *,
    warm_start: bool = True,
    adam_params: dict = None,  # safe defaults handled below
    lbfgs_params: dict = None,
    runtime_threads: int | None = None,  # NEW: per-process CPU threads; GPU workers force 1
):
    # --- per-worker thread policy (NO interop here) ---
    if _is_cuda_device(device):
        _set_torch_threads(1, interop_threads=None)  # GPU worker: minimal CPU threads
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))
    else:
        if runtime_threads is not None:
            _set_torch_threads(runtime_threads, interop_threads=None)  # CPU worker: set only intra-op

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

    obj = BLADE(logY, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, device=device)

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
                logger=run_log,
                phase="adam",
                outer_step=i,
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
                    logger=run_log,
                    phase="lbfgs",
                    outer_step=i,
                )
            except Exception as e:
                print(f"[WARN] ptimisation failed at iter {i} rep {Rep} ({e})]")

        obj.Update_Alpha_Group(Expected=Expected)

        if Update_SigmaY:
            obj.Update_SigmaY()

        with torch.no_grad():
            obj_val = float(obj.E_step(obj.Nu, obj.Beta, obj.Omega))
            obj_func[i] = obj_val
            if not np.isfinite(obj_val):
                print(f"[WARN] non-finite ELBO at outer iter {i} rep {Rep}; stopping.")
                obj_func = obj_func[: i + 1]
                break
            if abs(obj_func[i] - obj_func[i - 1]) < minDiff:
                obj_func = obj_func[: i + 1]
                break

    obj.Fix_par["Nu"] = False
    obj.Fix_par["Omega"] = True
    obj.Fix_par["Beta"] = True
    obj.Optimize(
        method="lbfgs",
        steps=12,
        lr=0.05,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
        logger=run_log,
        phase="polish:Nu",
        outer_step=iter,
    )

    obj.Fix_par["Nu"] = True
    obj.Fix_par["Omega"] = False
    obj.Fix_par["Beta"] = True
    obj.Optimize(
        method="lbfgs",
        steps=12,
        lr=0.05,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
        logger=run_log,
        phase="polish:Omega",
        outer_step=iter,
    )

    obj.Fix_par["Nu"] = True
    obj.Fix_par["Omega"] = True
    obj.Fix_par["Beta"] = False
    obj.Optimize(
        method="lbfgs",
        steps=12,
        lr=0.05,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
        logger=run_log,
        phase="polish:Beta",
        outer_step=iter,
    )

    obj.Fix_par["Nu"] = False
    obj.Fix_par["Omega"] = False
    obj.Fix_par["Beta"] = False

    with torch.no_grad():
        obj_func.append(float(obj.E_step(obj.Nu, obj.Beta, obj.Omega)))

    obj.train_log = getattr(obj, "train_log", []) + run_log.records
    return obj, obj_func, Rep


def _visible_cuda_devices():
    """Honor CUDA_VISIBLE_DEVICES if set; else enumerate real devices."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        toks = [t.strip() for t in vis.split(",") if t.strip() != ""]
        return [f"cuda:{i}" for i in range(len(toks))]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return []


def _free_vram_gb(dev_idx: int) -> float:
    """Free VRAM on a device in GiB; fail-open to inf on odd setups."""
    try:
        free_b, _ = torch.cuda.mem_get_info(dev_idx)
        return free_b / (1024**3)
    except Exception:
        return float("inf")


def _set_threads(num_threads: int, interop_threads: int = 1):
    """Set per-process CPU threading (PyTorch + BLAS)."""
    try:
        torch.set_num_threads(max(1, int(num_threads)))
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))


def plan_execution(
    Njob: int | None,
    Nrep: int,
    threads_per_job: int | None,
    backend: str = "auto",
    est_vram_gb: float | None = None,
    vram_soft_frac: float = 0.85,
):
    notes = []
    gpus = _visible_cuda_devices()
    ngpu = len(gpus)

    # pick backend
    if backend == "auto":
        backend = "gpu" if ngpu > 0 else "cpu"

    # ---------- CPU plan ----------
    if backend == "cpu" or ngpu == 0:
        backend = "cpu"
        cores = os.cpu_count() or 1
        # keep worker count modest by default
        n_jobs_eff = int(Njob) if Njob is not None else min(8, cores, Nrep)
        n_jobs_eff = max(1, min(n_jobs_eff, cores))
        # SAFE DEFAULT: 1 thread per worker unless user overrides
        t_per_job = int(threads_per_job) if threads_per_job is not None else 1
        devices = ["cpu"] * n_jobs_eff
        notes.append(f"CPU mode: {n_jobs_eff} workers × {t_per_job} threads (cores={cores}).")

    # ---------- GPU plan ----------
    else:
        backend = "gpu"
        n_jobs_target = int(Njob) if Njob is not None else min(ngpu, Nrep)
        if n_jobs_target <= ngpu:
            n_jobs_eff = max(1, min(n_jobs_target, ngpu, Nrep))
            devices = gpus[:n_jobs_eff]
            # SAFE DEFAULT: 1 thread per worker on GPU
            t_per_job = int(threads_per_job) if threads_per_job is not None else 1
            notes.append(f"GPU mode: {n_jobs_eff} workers over {ngpu} GPUs; threads/job={t_per_job}.")
        else:
            per_gpu_req = int(math.ceil(n_jobs_target / ngpu))
            devices = []
            for gi in range(ngpu):
                k = per_gpu_req
                if est_vram_gb:
                    free_gb = _free_vram_gb(gi) * vram_soft_frac
                    cap = max(1, int(free_gb // float(est_vram_gb)))
                    k = max(1, min(k, cap))
                devices.extend([gpus[gi]] * k)

            devices = devices[: min(n_jobs_target, Nrep)]
            n_jobs_eff = len(devices) if devices else min(ngpu, Nrep)
            if not devices:
                devices = gpus[:n_jobs_eff]
                notes.append("Packed plan soft-capped to 0 by VRAM; falling back to 1 per GPU.")
            # SAFE DEFAULT: 1 thread per worker on GPU
            t_per_job = int(threads_per_job) if threads_per_job is not None else 1
            notes.append(f"Packed GPU mode: {n_jobs_eff} workers across {ngpu} GPUs; threads/job={t_per_job}.")

    rep_device = lambda rep: devices[rep % max(1, len(devices))]
    return {
        "backend": backend,
        "devices": devices,
        "n_jobs_eff": n_jobs_eff,
        "threads_per_job_eff": t_per_job,
        "rep_device": rep_device,
        "notes": notes,
    }


# =========================
# Revised Framework_Iterative
# =========================


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
    warm_start: bool = True,
    collect_logs: bool = False,
    adam_params: dict = None,
    lbfgs_params: dict = None,
    backend: str = "auto",  # "auto" | "gpu" | "cpu"
    threads_per_job: int | None = None,  # per-process CPU threads
):
    # --- defaults for optimizer kwargs ---
    adam_params = adam_params or {"lr": 0.001, "steps": 200, "betas": [0.9, 0.98], "grad_clip": 10000.0}

    lbfgs_params = lbfgs_params or {"lr": 0.05, "steps": 12, "max_iter": 12, "history_size": 100}

    args = locals()

    Ngene, Nsample = Y.shape
    if Ind_Marker is None:
        Ind_Marker = [True] * Ngene

    X_small = X[Ind_Marker, :]
    Y_small = Y[Ind_Marker, :]
    stdX_small = stdX[Ind_Marker, :]

    # ---------- Plan execution (devices, concurrency, threads) ----------
    est_vram_hint = None
    try:
        est_vram_hint = (adam_params.get("est_vram_gb") if adam_params else None) or (
            lbfgs_params.get("est_vram_gb") if lbfgs_params else None
        )
    except Exception:
        pass

    plan = plan_execution(
        Njob=Njob,
        Nrep=Nrep,
        threads_per_job=threads_per_job,
        backend=backend,
        est_vram_gb=est_vram_hint,
        vram_soft_frac=0.85,
    )
    devices = plan["devices"]
    Njob_eff = plan["n_jobs_eff"]
    runtime_threads = plan["threads_per_job_eff"]
    rep_device = plan["rep_device"]

    for line in plan["notes"]:
        print("[Framework]", line)

    # Parent: set thread limits before any parallel work
    _set_threads(runtime_threads, interop_threads=1)

    # ---------- Now safe to do any threaded work in parent ----------
    print(f"start optimization using marker genes: {Y_small.shape[0]} genes out of {Ngene} genes.")
    print("Initialization with Support vector regression")

    # Use effective concurrency here too
    Init_Fraction, Ind_use = SVR_Initialization(X_small, Y_small, Njob=(Njob_eff or 1), Nus=[0.25, 0.5, 0.75])

    # ---------- worker wrapper (with OOM hint) ----------
    def _iter_one(rep):
        dev = rep_device(rep)
        # print(f"[rep {rep:02d}] device={dev}, threads={runtime_threads}")
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
                runtime_threads=runtime_threads,  # worker sets *only* intra-op
            )
        except RuntimeError as e:
            if (
                "out of memory" in str(e).lower()
                and plan["backend"] == "gpu"
                and len(set(devices)) == 1  # packed single-GPU case
                and Njob_eff > 1
            ):
                print(
                    f"[rep {rep:02d}] CUDA OOM in packed single-GPU mode. "
                    f"Try smaller batch / fewer workers (Njob) / threads_per_job=1."
                )
            raise

    # print("DEBUG torch.cuda.is_available:", torch.cuda.is_available())
    # print("DEBUG visible devices:", _visible_cuda_devices())
    # print("DEBUG env CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # print("DEBUG plan backend/devices/Njob_eff/threads:",
    #     plan["backend"], plan["devices"], plan["n_jobs_eff"], plan["threads_per_job_eff"])
    # ---------- Execute (parallel/serial based on Njob_eff) ----------
    if Temperature is None or Temperature is False:
        with parallel_backend("loky", n_jobs=Njob_eff):
            triples = Parallel(n_jobs=Njob_eff, verbose=10)(delayed(_iter_one)(rep) for rep in range(Nrep))
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
        out = outs[best]
        conv = convs[best]

    else:
        # --- temperature schedule branch (serial here) ---
        if Temperature is True:
            Temperature = [1, 100]
        else:
            if len(Temperature) != 2:
                raise ValueError("Temperature must be None, True, or [Tmin, Tmax].")
            if Temperature[1] < Temperature[0]:
                raise ValueError("Max temperature must be ≥ min temperature.")

        triples = [
            Iterative_Optimization(
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
                TempRange=np.linspace(Temperature[0], Temperature[1], num=IterMax),
                Update_SigmaY=Update_SigmaY,
                warm_start=warm_start,
                adam_params=adam_params,
                lbfgs_params=lbfgs_params,
                runtime_threads=runtime_threads,
            )
            for rep in range(Nrep)
        ]

        outs, convs, Reps = zip(*triples)
        with torch.no_grad():
            cri = [float(obj.E_step(obj.Nu, obj.Beta, obj.Omega)) for obj in outs]
        best = int(np.nanargmax(cri))
        out = outs[best]
        conv = convs[best]

    if collect_logs:
        logs = [{"rep": r, "log": getattr(o, "train_log", [])} for o, r in zip(outs, Reps)]
        return out, conv, zip(outs, cri), args, logs

    return out, conv, zip(outs, cri), args


#########NUMBA functions for purification########


@njit(fastmath=True)
def ExpF_numba(Beta, Ncell):
    # NSample by Ncell (Expectation of F)
    output = np.empty(Beta.shape)
    for c in range(Ncell):
        output[:, c] = Beta[:, c] / np.sum(Beta, axis=1)
    return output


@njit(fastmath=True)
def ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Expected value of Y)
    ExpB = ExpF_numba(Beta, Ncell)  # Nsample by Ncell
    out = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            out[:, i] = out[:, i] + ExpB[i, c] * np.exp(Nu[i, :, c] + 0.5 * np.square(Omega[:, c]))

    return out


@njit(fastmath=True)
def VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample):
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1)  # Nsample
    Btilda = ExpF_numba(Beta, Ncell)  # Nsample by Ncell
    VarB = Btilda * (1 - Btilda)
    for c in range(Ncell):
        VarB[:, c] = VarB[:, c] / (B0 + 1)

    # Nsample Ncell Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:, l, k] = -Btilda[:, l] * Btilda[:, k] / (1 + B0)

    # Ngene by Nsample by Ncell by Ncell
    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:, i, l, k] = np.exp(
                    Nu[i, :, k] + Nu[i, :, l] + 0.5 * (np.square(Omega[:, k]) + np.square(Omega[:, l]))
                )

    VarTerm = np.zeros((Ngene, Nsample))
    for i in range(Nsample):
        for c in range(Ncell):
            VarTerm[:, i] = (
                VarTerm[:, i]
                + np.exp(2 * Nu[i, :, c] + 2 * np.square(Omega)[:, c]) * (VarB[i, c] + np.square(Btilda[i, c]))
                - np.exp(2 * Nu[i, :, c] + np.square(Omega[:, c])) * (np.square(Btilda[i, c]))
            )

    # Ngene by Ncell
    CovTerm = np.zeros((Ngene, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:, i] = CovTerm[:, i] + CovX[:, i, l, k] * CovB[i, l, k]

    return VarTerm + CovTerm


@njit(fastmath=True)
def Estep_PY_numba(Y, SigmaY, Nu, Omega, Beta, Ngene, Ncell, Nsample):
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)

    a = Var / Exp / Exp

    return np.sum(-0.5 / np.square(SigmaY) * (a + np.square((Y - np.log(Exp)) - 0.5 * a)))


@njit(fastmath=True)
def Estep_PX_numba(Mu0, Nu, Omega, Alpha0, Beta0, Kappa0, Ncell, Nsample):
    NuExp = np.sum(Nu, 0) / Nsample  # expected Nu, Ngene by Ncell
    AlphaN = Alpha0 + 0.5 * Nsample  # Posterior Alpha

    ExpBetaN = (
        Beta0
        + (Nsample - 1) / 2 * np.square(Omega)
        + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (np.square(Omega) / Nsample + np.square(NuExp - Mu0))
    )

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5 * np.square(Nu[i, :, :] - NuExp)

    return np.sum(-AlphaN * np.log(ExpBetaN))


@njit(fastmath=True)
def grad_Nu_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # return Nsample by Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0) / Nsample

    Diff = np.zeros((Ngene, Ncell))
    ExpBetaN = (
        Beta0
        + (Nsample - 1) / 2 * np.square(Omega)
        + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (np.square(Omega) / Nsample + np.square(NuExp - Mu0))
    )

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5 * np.square(Nu[i, :, :] - NuExp)
        Diff = Diff + (Nu[i, :, :] - NuExp) / Nsample

    Nominator = np.empty((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nominator[i, :, :] = Nu[i, :, :] - NuExp - Diff + Kappa0 / (Kappa0 + Nsample) * (NuExp - Mu0)

    grad_PX = -AlphaN * Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1)  # Nsample
    Btilda = ExpF_numba(Beta, Ncell)  # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:, l, k] = -Btilda[:, l] * Btilda[:, k] / (1 + B0)

    ExpX = np.empty(Nu.shape)  # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i, :, :] = np.exp(Nu[i, :, :] + 0.5 * np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:, i, l, k] = ExpX[i, :, l] * ExpX[i, :, k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:, l, i] = CovTerm[:, l, i] + 2 * CovX[:, i, l, k] * CovB[i, l, k]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:, c, i] = ExpX[i, :, c] * Btilda[i, c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.empty(Nu.shape)
    for i in range(Nsample):
        VarX[i, :, :] = np.exp(2 * Nu[i, :, :] + 2 * np.square(Omega))

    VarB = Btilda * (1 - Btilda)
    for c in range(Ncell):
        VarB[:, c] = VarB[:, c] / (B0 + 1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:, c, i] = 2 * VarX[i, :, c] * (VarB[i, c] + np.square(Btilda[i, c])) - 2 * CovX[
                :, i, c, c
            ] * np.square(Btilda[i, c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:, c, :] = (g_Var[:, c, :] - 2 * g_Exp[:, c, :] / Exp * Var) / np.power(Exp, 2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2 * np.square(Exp))
    for c in range(Ncell):
        b[:, c, :] = -Diff * (2 * g_Exp[:, c, :] / Exp + a[:, c, :])

    grad_PY = np.zeros((Nsample, Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:, :, c] = -np.transpose(0.5 / np.square(SigmaY) * (a[:, c, :] + b[:, c, :]))

    return grad_PX * (1 / weight) + grad_PY


@njit(fastmath=True)
def grad_Omega_numba(Y, SigmaY, Nu, Omega, Beta, Mu0, Alpha0, Beta0, Kappa0, Ngene, Ncell, Nsample, weight):
    # Ngene by Ncell

    # gradient of PX (first term)
    AlphaN = Alpha0 + Nsample * 0.5
    NuExp = np.sum(Nu, 0) / Nsample
    ExpBetaN = (
        Beta0
        + (Nsample - 1) / 2 * np.square(Omega)
        + Kappa0 * Nsample / (2 * (Kappa0 + Nsample)) * (np.square(Omega) / Nsample + np.square(NuExp - Mu0))
    )

    for i in range(Nsample):
        ExpBetaN = ExpBetaN + 0.5 * np.square(Nu[i, :, :] - NuExp)

    Nominator = -AlphaN * (Nsample - 1) * Omega + Kappa0 / (Kappa0 + Nsample) * Omega
    grad_PX = Nominator / ExpBetaN

    # gradient of PY (second term)
    # Ngene by Nsample (Variance value of Y)
    B0 = np.sum(Beta, axis=1)  # Nsample
    Btilda = ExpF_numba(Beta, Ncell)  # Nsample by Ncell

    # Ngene by Ncell by Nsample
    Exp = ExpQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample
    Var = VarQ_numba(Nu, Beta, Omega, Ngene, Ncell, Nsample)  # Ngene by Nsample

    # Ngene by Nsample by Ncell by Ncell
    CovB = np.empty((Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            CovB[:, l, k] = -Btilda[:, l] * Btilda[:, k] / (1 + B0)

    ExpX = np.exp(Nu)  # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX[i, :, :] = ExpX[i, :, :] * np.exp(0.5 * np.square(Omega))

    CovX = np.empty((Ngene, Nsample, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[:, i, l, k] = ExpX[i, :, l] * ExpX[i, :, k]

    # Ngene by Ncell by Nsample
    CovTerm = np.zeros((Ngene, Ncell, Nsample))
    for l in range(Ncell):
        for k in range(Ncell):
            if l != k:
                for i in range(Nsample):
                    CovTerm[:, l, i] = CovTerm[:, l, i] + 2 * CovX[:, i, l, k] * CovB[i, l, k] * Omega[:, l]

    # Ngene by Ncell by Nsample
    g_Exp = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        for i in range(Nsample):
            g_Exp[:, c, i] = ExpX[i, :, c] * Btilda[i, c] * Omega[:, c]

    # Ngene by Ncell by Nsample
    g_Var = np.empty((Ngene, Ncell, Nsample))
    VarX = np.exp(2 * Nu)
    for i in range(Nsample):
        VarX[i, :, :] = VarX[i, :, :] * np.exp(2 * np.square(Omega))

    VarB = Btilda * (1 - Btilda)
    for c in range(Ncell):
        VarB[:, c] = VarB[:, c] / (B0 + 1)

    for c in range(Ncell):
        for i in range(Nsample):
            g_Var[:, c, i] = 4 * Omega[:, c] * VarX[i, :, c] * (VarB[i, c] + np.square(Btilda[i, c])) - 2 * Omega[
                :, c
            ] * CovX[:, i, c, c] * np.square(Btilda[i, c])
    g_Var = g_Var + CovTerm

    # Ngene by Ncell by Nsample
    a = np.empty((Ngene, Ncell, Nsample))
    for c in range(Ncell):
        a[:, c, :] = (g_Var[:, c, :] - 2 * g_Exp[:, c, :] * Var / Exp) / np.power(Exp, 2)

    b = np.empty((Ngene, Ncell, Nsample))
    Diff = Y - np.log(Exp) - Var / (2 * np.square(Exp))
    for c in range(Ncell):
        b[:, c, :] = -Diff * (2 * g_Exp[:, c, :] / Exp + a[:, c, :])

    grad_PY = np.zeros((Ngene, Ncell))
    for c in range(Ncell):
        grad_PY[:, c] = np.sum(-0.5 / np.square(SigmaY) * (a[:, c, :] + b[:, c, :]), axis=1)

    # Q(X) (fourth term)
    grad_QX = -Nsample / Omega

    return grad_PX * (1 / weight) + grad_PY - grad_QX * (1 / weight)


@njit(fastmath=True)
def g_Exp_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    ExpX = np.exp(Nu)
    for i in range(Nsample):
        ExpX[i, :, :] = ExpX[i, :, :] * np.exp(0.5 * np.square(Omega))  # Nsample by Ngene by Ncell
    B0mat = np.empty(Beta.shape)
    for c in range(Ncell):
        B0mat[:, c] = Beta[:, c] / np.square(B0)

    tmp = np.empty((Nsample, Ngene))
    tExpX = np.ascontiguousarray(ExpX.transpose(0, 2, 1))  ## Make tExpX contiguous again
    for i in range(Nsample):
        tmp[i, :] = np.dot(B0mat[i, :], tExpX[i, ...])
    B0mat = tmp

    g_Exp = np.empty((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Exp[s, c, :] = t(ExpX[s, :, c] / B0[s]) - B0mat[s, :]

    return g_Exp


@njit(fastmath=True)
def g_Var_Beta_numba(Nu, Omega, Beta, B0, Ngene, Ncell, Nsample):
    B0Rep = np.empty(Beta.shape)  # Nsample by Ncell
    for c in range(Ncell):
        B0Rep[:, c] = B0

    aa = (B0Rep - Beta) * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aa = aa / (np.power(B0Rep, 3) * np.square(B0Rep + 1))
    aa = aa + 2 * Beta * (B0Rep - Beta) / np.power(B0Rep, 3)

    aaNotT = Beta * B0Rep * (B0Rep + 1) - (3 * B0Rep + 2) * Beta * (B0Rep - Beta)
    aaNotT = aaNotT / (np.power(B0Rep, 3) * np.square(B0Rep + 1))
    aaNotT = aaNotT + 2 * Beta * (0 - Beta) / np.power(B0Rep, 3)

    ExpX2 = 2 * Nu  # Nsample by Ngene by Ncell
    for i in range(Nsample):
        ExpX2[i, :, :] = np.exp(ExpX2[i, :, :] + 2 * np.square(Omega))

    g_Var = np.zeros((Nsample, Ncell, Ngene))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s, c, :] = t(ExpX2[s, :, c]) * aa[s, c]

    for i in range(Ncell):
        for j in range(Ncell):
            if i != j:
                for s in range(Nsample):
                    g_Var[s, i, :] = g_Var[s, i, :] + t(ExpX2[s, :, j]) * aaNotT[s, j]

    B_B02 = Beta / np.square(B0Rep)  # Beta / (Beta0^2) / Nsample by Ncell
    B0B0_1 = B0Rep * (B0Rep + 1)  # Beta0 (Beta0+1) / Nsample by Nell
    B2_B03 = np.square(Beta) / np.power(B0Rep, 3)  # Beta^2 / (Beta0^3) / Nsample by Ncell

    ExpX = np.empty(Nu.shape)
    for i in range(Nsample):
        ExpX[i, :, :] = np.exp(2 * Nu[i, :, :] + np.square(Omega))

    for s in range(Nsample):
        for c in range(Ncell):
            g_Var[s, c, :] = g_Var[s, c, :] - 2 * t(ExpX[s, :, c]) * B_B02[s, c]

    Dot = np.zeros((Nsample, Ngene))
    for i in range(Nsample):
        for c in range(Ncell):
            Dot[i, :] = Dot[i, :] + B2_B03[i, c] * ExpX[i, :, c]

    for c in range(Ncell):
        g_Var[:, c, :] = g_Var[:, c, :] + 2 * Dot

    # Ngene by Nsample by Ncell by N cell
    ExpX = np.empty((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        ExpX[i, :, :] = np.exp(Nu[i, :, :] + 0.5 * np.square(Omega))
    CovX = np.empty((Nsample, Ngene, Ncell, Ncell))
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                CovX[i, :, l, k] = ExpX[i, :, l] * ExpX[i, :, k]

    gradCovB = np.empty((Nsample, Ncell, Ncell))
    B03_2_B03_B0_1 = (3 * B0 + 2) / np.power(B0, 3) / np.square(B0 + 1)
    for l in range(Ncell):
        for k in range(Ncell):
            gradCovB[:, l, k] = Beta[:, l] * Beta[:, k] * B03_2_B03_B0_1

    # Nsample by Ncell by Ncell by Ngene
    CovTerm1 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    CovTerm2 = np.zeros((Nsample, Ncell, Ncell, Ngene))
    B_B0_1_B0B0_1 = Beta * (B0Rep + 1) / np.square(B0B0_1)  # Nsample by Ncell
    for l in range(Ncell):
        for k in range(Ncell):
            for i in range(Nsample):
                if l != k:
                    CovTerm1[i, l, k, :] = gradCovB[i, l, k] * CovX[i, :, l, k]
                    CovTerm2[i, l, k, :] = B_B0_1_B0B0_1[i, l] * CovX[i, :, l, k]

    for c in range(Ncell):
        g_Var[:, c, :] = g_Var[:, c, :] + np.sum(np.sum(CovTerm1, axis=1), axis=1)
    g_Var = g_Var - 2 * np.sum(CovTerm2, axis=1)

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
        a[:, c, :] = np.divide((g_Var[:, c, :] * t(Exp) - 2 * g_Exp[:, c, :] * t(Var)), np.power(t(Exp), 3))

    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2 * np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s, c, g] = -(Y[g, s] - np.log(Exp[g, s]) - Var_Exp2[g, s]) * (
                    2 * np.divide(g_Exp[s, c, g], Exp[g, s]) + a[s, c, g]
                )

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s, c] = grad_PY[s, c] - np.sum(0.5 / np.square(SigmaY[:, s]) * (a[s, c, :] + b[s, c, :]))

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
        a[:, c, :] = np.divide((g_Var[:, c, :] * t(Exp) - 2 * g_Exp[:, c, :] * t(Var)), np.power(t(Exp), 3))

    b = np.empty((Nsample, Ncell, Ngene))
    Var_Exp2 = np.divide(Var, 2 * np.square(Exp))
    for s in range(Nsample):
        for c in range(Ncell):
            for g in range(Ngene):
                b[s, c, g] = -(Y[g, s] - np.log(Exp[g, s]) - Var_Exp2[g, s]) * (
                    2 * np.divide(g_Exp[s, c, g], Exp[g, s]) + a[s, c, g]
                )

    grad_PY = np.zeros((Nsample, Ncell))
    for s in range(Nsample):
        for c in range(Ncell):
            grad_PY[s, c] = grad_PY[s, c] - np.sum(0.5 / np.square(SigmaY[:, s]) * (a[s, c, :] + b[s, c, :]))

    return grad_PY


######Casting function#######


def to_torch(array, device="cuda"):
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
    ):
        self.weight = 1

        # Ensure all tensor inputs are converted to numpy arrays
        self.Y = ensure_numpy(Y)
        self.Ngene, self.Nsample = self.Y.shape

        self.Mu0 = ensure_numpy(Mu0) if isinstance(Mu0, np.ndarray) else np.zeros((self.Ngene, Mu0))
        self.Ncell = self.Mu0.shape[1]

        self.SigmaY = (
            ensure_numpy(SigmaY) if isinstance(SigmaY, np.ndarray) else np.ones((self.Ngene, self.Nsample)) * SigmaY
        )
        self.Alpha = (
            ensure_numpy(Alpha) if isinstance(Alpha, np.ndarray) else np.ones((self.Nsample, self.Ncell)) * Alpha
        )
        self.Omega = (
            ensure_numpy(Omega_Init)
            if isinstance(Omega_Init, np.ndarray)
            else np.zeros((self.Ngene, self.Ncell)) + Omega_Init
        )
        self.Nu = ensure_numpy(Nu_Init) if Nu_Init is not None else np.zeros((self.Nsample, self.Ngene, self.Ncell))
        self.Beta = (
            ensure_numpy(Beta_Init) if isinstance(Beta_Init, np.ndarray) else np.ones((self.Nsample, self.Ncell))
        )
        self.Alpha0 = (
            ensure_numpy(Alpha0) if isinstance(Alpha0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Alpha0
        )
        self.Beta0 = ensure_numpy(Beta0) if isinstance(Beta0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Beta0
        self.Kappa0 = (
            ensure_numpy(Kappa0) if isinstance(Kappa0, np.ndarray) else np.ones((self.Ngene, self.Ncell)) * Kappa0
        )

        self.Fix_par = {"Beta": fix_Beta, "Nu": fix_Nu, "Omega": fix_Omega}

    def Ydiff(self, Nu, Beta):
        F = self.ExpF(Beta)
        Ypred = np.dot(np.exp(Nu), t(F))
        return np.sum(np.square(self.Y - Ypred))

    def ExpF_numba(self, Beta):
        # NSample by Ncell (Expectation of F)
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
        return -(np.sum(loggamma(self.Alpha)) - np.sum(loggamma(self.Alpha.sum(axis=1)))) + np.sum(
            (self.Alpha - 1) * (digamma(Beta) - np.tile(digamma(np.sum(Beta, axis=1))[:, np.newaxis], [1, self.Ncell]))
        )

    # Expectation of log Q(X)
    def Estep_QX(self, Omega):
        return -self.Nsample * np.sum(np.log(Omega))

    # Expectation of log Q(F)
    def Estep_QF(self, Beta):
        return -(np.sum(loggamma(Beta)) - np.sum(loggamma(Beta.sum(axis=1)))) + np.sum(
            (Beta - 1) * (digamma(Beta) - np.tile(digamma(np.sum(Beta, axis=1))[:, np.newaxis], [1, self.Ncell]))
        )

    def grad_Nu(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Nu_numba(
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

    def grad_Omega(self, Nu, Omega, Beta):
        # return Ngene by Ncell
        return grad_Omega_numba(
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

    def g_Exp_Beta(self, Nu, Omega, Beta, B0):
        return g_Exp_Beta_numba(Nu, Omega, Beta, B0, self.Ngene, self.Ncell, self.Nsample)

    def grad_Beta(self, Nu, Omega, Beta):
        # return Nsample by Ncell
        B0 = np.sum(self.Beta, axis=1)

        grad_PY = g_PY_Beta(Nu, Beta, Omega, self.Y, self.SigmaY, B0, self.Ngene, self.Ncell, self.Nsample)

        grad_PF = (self.Alpha - 1) * polygamma(1, Beta) - np.tile(
            np.sum((self.Alpha - 1) * np.tile(polygamma(1, B0)[:, np.newaxis], [1, self.Ncell]), axis=1)[:, np.newaxis],
            [1, self.Ncell],
        )

        grad_QF = (Beta - 1) * polygamma(1, Beta) - np.tile(
            np.sum((Beta - 1) * np.tile(polygamma(1, B0)[:, np.newaxis], [1, self.Ncell]), axis=1)[:, np.newaxis],
            [1, self.Ncell],
        )

        return grad_PY + grad_PF * np.sqrt(self.Ngene / self.Ncell) - grad_QF * np.sqrt(self.Ngene / self.Ncell)

    # E step
    def E_step(self, Nu, Beta, Omega):
        PX = self.Estep_PX(Nu, Omega) * (1 / self.weight)
        PY = self.Estep_PY(Nu, Omega, Beta)
        PF = self.Estep_PF(Beta) * np.sqrt(self.Ngene / self.Ncell)
        QX = self.Estep_QX(Omega) * (1 / self.weight)
        QF = self.Estep_QF(Beta) * np.sqrt(self.Ngene / self.Ncell)

        return PX + PY + PF - QX - QF

    def Optimize(self):
        # loss function
        def loss(params):
            Nu = params[0 : self.Ncell * self.Ngene * self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[
                self.Ncell * self.Ngene * self.Nsample : (
                    self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell
                )
            ].reshape(self.Ngene, self.Ncell)
            Beta = params[
                (self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell) : (
                    self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell + self.Nsample * self.Ncell
                )
            ].reshape(self.Nsample, self.Ncell)

            if self.Fix_par["Nu"]:
                Nu = self.Nu
            if self.Fix_par["Beta"]:
                Beta = self.Beta
            if self.Fix_par["Omega"]:
                Omega = self.Omega

            return -self.E_step(Nu, Beta, Omega)

        # gradient function
        def grad(params):
            Nu = params[0 : self.Ncell * self.Ngene * self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
            Omega = params[
                self.Ncell * self.Ngene * self.Nsample : (
                    self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell
                )
            ].reshape(self.Ngene, self.Ncell)
            Beta = params[
                (self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell) : (
                    self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell + self.Nsample * self.Ncell
                )
            ].reshape(self.Nsample, self.Ncell)

            if self.Fix_par["Nu"]:
                g_Nu = np.zeros(Nu.shape)
            else:
                g_Nu = -self.grad_Nu(Nu, Omega, Beta)

            if self.Fix_par["Omega"]:
                g_Omega = np.zeros(Omega.shape)
            else:
                g_Omega = -self.grad_Omega(Nu, Omega, Beta)

            if self.Fix_par["Beta"]:
                g_Beta = np.zeros(Beta.shape)
            else:
                g_Beta = -self.grad_Beta(Nu, Omega, Beta)

            g = np.concatenate((g_Nu.flatten(), g_Omega.flatten(), g_Beta.flatten()))

            return g

        # Perform Optimization
        Init = np.concatenate((self.Nu.flatten(), self.Omega.flatten(), self.Beta.flatten()))
        bounds = [
            (-np.inf, np.inf) if i < (self.Ncell * self.Ngene * self.Nsample) else (0.0000001, 100)
            for i in range(len(Init))
        ]

        out = scipy.optimize.minimize(
            fun=loss, x0=Init, bounds=bounds, jac=grad, options={"disp": False}, method="L-BFGS-B"
        )

        params = out.x

        self.Nu = params[0 : self.Ncell * self.Ngene * self.Nsample].reshape(self.Nsample, self.Ngene, self.Ncell)
        self.Omega = params[
            self.Ncell * self.Ngene * self.Nsample : (self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell)
        ].reshape(self.Ngene, self.Ncell)
        self.Beta = params[
            (self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell) : (
                self.Ncell * self.Ngene * self.Nsample + self.Ngene * self.Ncell + self.Nsample * self.Ncell
            )
        ].reshape(self.Nsample, self.Ncell)

        self.log = out.success

    # Reestimation of Nu at specific weight
    def Reestimate_Nu(self, weight=100):
        self.weight = weight
        self.Optimize()
        return self

    def Check_health(self):
        # check if optimization is done
        if not hasattr(self, "log"):
            warnings.warn("No optimization is not done yet", Warning, stacklevel=2)

        # check values in hyperparameters
        if not np.all(np.isfinite(self.Y)):
            warnings.warn("non-finite values detected in bulk gene expression data (Y).", Warning, stacklevel=2)

        if np.any(self.Y < 0):
            warnings.warn(
                "Negative expression levels were detected in bulk gene expression data (Y).", Warning, stacklevel=2
            )

        if np.any(self.Alpha <= 0):
            warnings.warn("Zero or negative values in Alpha", Warning, stacklevel=2)

        if np.any(self.Beta <= 0):
            warnings.warn("Zero or negative values in Beta", Warning, stacklevel=2)

        if np.any(self.Alpha0 <= 0):
            warnings.warn("Zero or negative values in Alpha0", Warning, stacklevel=2)

        if np.any(self.Beta0 <= 0):
            warnings.warn("Zero or negative values in Beta0", Warning, stacklevel=2)

        if np.any(self.Kappa0 <= 0):
            warnings.warn("Zero or negative values in Kappa0", Warning, stacklevel=2)

    def Update_Alpha(self, Expected=None, Temperature=None):  # if Expected fraction is given, that part will be fixed
        # Updating Alpha
        Fraction = self.ExpF(self.Beta)
        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if (
                    "Group" in Expected
                ):  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected["Group"]
                else:
                    Group = np.identity(Expected["Expectation"].shape[1])
                Expected = Expected["Expectation"]
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError("Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)")

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                IndG = np.where(~np.isnan(Expected[sample, :]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group, :] == 1)[0]
                    Fraction[sample, IndCell] = Fraction[sample, IndCell] / np.sum(
                        Fraction[sample, IndCell]
                    )  # make fraction sum to one for the group
                    Fraction[sample, IndCell] = (
                        Fraction[sample, IndCell] * Expected[sample, group]
                    )  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[sample, IndNan] = Fraction[sample, IndNan] / np.sum(
                    Fraction[sample, IndNan]
                )  # normalize the rest of cell types (sum to one)
                Fraction[sample, IndNan] = Fraction[sample, IndNan] * (
                    1 - np.sum(Expected[sample, IndG])
                )  # assign determined fraction for the rest of cell types

        if Temperature is not None:
            self.Alpha = Temperature * Fraction
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample, :] = Fraction[sample, :] * np.sum(self.Beta[sample, :])

    def Update_Alpha_Group(
        self, Expected=None, Temperature=None
    ):  # if Expected fraction is given, that part will be fixed
        # Updating Alpha
        AvgBeta = np.mean(self.Beta, 0)
        Fraction_Avg = AvgBeta / np.sum(AvgBeta)

        if Expected is not None:  # Reflect the expected values
            # expectaion can be a diction (with two keys; Group and Expectation) or just a matrix
            if type(Expected) is dict:
                if (
                    "Group" in Expected
                ):  # Group (Ngroup by Nctype matrix) indicates a group of cell types with known collective fraction
                    Group = Expected["Group"]
                else:
                    Group = np.identity(Expected["Expectation"].shape[1])
                Expected = Expected["Expectation"]
            else:
                Group = np.identity(Expected.shape[1])

            if self.Beta.shape[0] != Expected.shape[0] or self.Beta.shape[1] != Group.shape[1]:
                raise ValueError("Pre-determined fraction is in wrong shape (should be Nsample by Ncelltype)")

            # rescale the fraction to meet the expected fraction
            for sample in range(self.Nsample):
                Fraction = np.copy(Fraction_Avg)
                IndG = np.where(~np.isnan(Expected[sample, :]))[0]
                IndCells = []

                for group in IndG:
                    IndCell = np.where(Group[group, :] == 1)[0]
                    Fraction[IndCell] = Fraction[IndCell] / np.sum(
                        Fraction[IndCell]
                    )  # make fraction sum to one for the group
                    Fraction[IndCell] = (
                        Fraction[IndCell] * Expected[sample, group]
                    )  # assign determined fraction for the group
                    IndCells = IndCells + list(IndCell)

                IndNan = np.setdiff1d(np.array(range(Group.shape[1])), np.array(IndCells))
                Fraction[IndNan] = Fraction[IndNan] / np.sum(
                    Fraction[IndNan]
                )  # normalize the rest of cell types (sum to one)
                Fraction[IndNan] = Fraction[IndNan] * (
                    1 - np.sum(Expected[sample, IndG])
                )  # assign determined fraction for the rest of cell types

                AlphaSum = np.sum(AvgBeta[IndNan]) / np.sum(Fraction[IndNan])
                self.Alpha[sample, :] = Fraction * AlphaSum
        else:
            for sample in range(self.Nsample):
                self.Alpha[sample, :] = AvgBeta

    def Update_SigmaY(self, SampleSpecific=False):
        Var = VarQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)
        Exp = ExpQ_numba(self.Nu, self.Beta, self.Omega, self.Ngene, self.Ncell, self.Nsample)

        a = Var / Exp / Exp
        b = np.square((self.Y - np.log(Exp)) - 0.5 * a)

        if SampleSpecific:
            self.SigmaY = np.sqrt(a + b)
        else:  # shared in all samples
            self.SigmaY = np.tile(np.mean(np.sqrt(a + b), axis=1)[:, np.newaxis], [1, self.Nsample])


def Parallel_Purification(obj, weight, iter=1000, minDiff=10e-4, Update_SigmaY=False):
    obj.Check_health()
    obj_func = [float("nan")] * iter
    obj_func[0] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)
    for i in range(1, iter):
        obj.Reestimate_Nu(weight=weight)
        if Update_SigmaY:
            obj.Update_SigmaY()
        obj_func[i] = obj.E_step(obj.Nu, obj.Beta, obj.Omega)

        # Check for convergence
        if np.abs(obj_func[i] - obj_func[i - 1]) < minDiff:
            break
    return obj, obj_func


def Purify_AllGenes(BLADE_object, Mu, Omega, Y, Ncores, Weight=100, sY=1, Alpha0=1000, Kappa0=1):
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
    logY = np.log(Y + 1)
    SigmaY = np.tile(np.std(logY, 1)[:, np.newaxis], [1, Nsample]) * sY + 0.1
    Beta0 = Alpha0 * np.square(Omega)
    Nu_Init = np.zeros((Nsample, Ngene, Ncell))
    for i in range(Nsample):
        Nu_Init[i, :, :] = Mu

    # Fetch objs per gene
    Ngene_total = Mu.shape[0]
    objs = []
    for ix in range(Ngene_total):
        objs.append(
            BLADE_numba(
                Y=np.atleast_2d(logY[ix, :]),
                SigmaY=np.atleast_2d(SigmaY[ix, :]),
                Mu0=np.atleast_2d(Mu[ix, :]),
                Alpha=obj.Alpha,
                Alpha0=Alpha0,
                Beta0=np.atleast_2d(Beta0[ix, :]),
                Kappa0=Kappa0,
                Nu_Init=np.reshape(np.atleast_3d(Nu_Init[:, ix, :]), (Nsample, 1, Ncell)),
                Omega_Init=np.atleast_2d(Omega[ix, :]),
                Beta_Init=obj.Beta,
                fix_Beta=True,
            )
        )

    outs = Parallel(n_jobs=Ncores, verbose=10)(delayed(Parallel_Purification)(obj, Weight) for obj in objs)

    objs, obj_func = zip(*outs)
    ## sum ofv over all genes
    obj_func = np.sum(obj_func, axis=0)
    logs = []
    ## Combine results from all genes
    for i, obj in enumerate(objs):
        logs.append(obj.log)
        if i == 0:
            Y = objs[0].Y
            SigmaY = objs[0].SigmaY
            Mu0 = objs[0].Mu0
            Alpha = objs[0].Alpha
            Alpha0 = objs[0].Alpha0
            Beta0 = objs[0].Beta0
            Kappa0 = objs[0].Kappa0
            Nu_Init = objs[0].Nu
            Omega_Init = objs[0].Omega
            Beta_Init = objs[0].Beta
        else:
            Y = np.concatenate((Y, obj.Y))
            SigmaY = np.concatenate((SigmaY, obj.SigmaY))
            Mu0 = np.concatenate((Mu0, obj.Mu0))
            Alpha0 = np.concatenate((Alpha0, obj.Alpha0))
            Beta0 = np.concatenate((Beta0, obj.Beta0))
            Kappa0 = np.concatenate((Kappa0, obj.Kappa0))
            Nu_Init = np.concatenate((Nu_Init, obj.Nu), axis=1)
            Omega_Init = np.concatenate((Omega_Init, obj.Omega))

    ## Create final merged BLADE obj to return
    obj = BLADE_numba(Y, SigmaY, Mu0, Alpha, Alpha0, Beta0, Kappa0, Nu_Init, Omega_Init, Beta_Init, fix_Beta=True)
    obj.log = logs

    return obj
