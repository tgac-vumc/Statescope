#!/usr/bin/env python3
"""
Statescope - Deconvolution Runner (HPC-friendly), with forced CP10K bulk normalization.

Input modes (choose exactly ONE of A/B/C/D):
A) --model-path <pickle>
B) --adata <.h5ad>  --bulk-csv <csv/tsv>  --celltype-key <obs col>
C) --signature-csv <csv/tsv>  --bulk-csv <csv/tsv>
D) --tumor-type <name>  --ncelltypes <int>  --bulk-csv <csv/tsv>

Optional:
--expectation-csv <csv/tsv>    (rows=samples, cols=subset of cell types)

Behavior:
  • Bulk is ALWAYS normalized to counts-per-10k (CP10K) immediately after reading.
  • Delimiter for CSV/TSV auto-detected (comma or tab).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Repo src path
THIS_DIR = os.path.dirname(__file__)
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import Statescope API
from Statescope.Statescope import Initialize_Statescope, Statescope
from BLADE_Deconvolution.BLADE import BLADE


# ------------------------- CLI ------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Run Statescope deconvolution (CP10K bulk normalization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Exactly ONE of these:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model-path', type=str,
                   help='Existing saved Statescope model (.pickle/.pkl).')
    g.add_argument('--adata', type=str,
                   help='Signature AnnData (.h5ad). Requires --bulk-csv and --celltype-key.')
    g.add_argument('--signature-csv', type=str,
                   help='Signature matrix CSV/TSV (genes x celltypes). Requires --bulk-csv.')
    g.add_argument('--tumor-type', type=str,
                   help='Built-in signature (e.g., PBMC). Requires --ncelltypes and --bulk-csv.')

    # Shared inputs
    p.add_argument('--bulk-csv', type=str,
                   help='Bulk expression CSV/TSV (genes x samples).')
    p.add_argument('--celltype-key', type=str, default='celltype',
                   help='AnnData .obs column with cell-type labels (only with --adata).')
    p.add_argument('--ncelltypes', type=int,
                   help='Number of cell types for --tumor-type mode (e.g., 7).')

    # Optional Expectation
    p.add_argument('--expectation-csv', type=str, default=None,
                   help='CSV/TSV: rows=samples, cols=subset of cell types; values in (0,1).')

    # Backend/parallel basics
    p.add_argument('--backend', choices=['auto','cpu','gpu'], default='auto',
                   help='Device backend.')
    p.add_argument('--nrep', type=int, default=10, help='Random restarts.')
    p.add_argument('--njobs', type=int, default=10, help='Parallel jobs (processes).')
    p.add_argument('--iter-max', type=int, default=1000, help='Max outer iterations.')

    # Warm-start toggle (passed through to Deconvolution)
    p.add_argument('--warm-start', action='store_true',
                   help='Enable Adam warm-up before L-BFGS.')
    p.add_argument('--no-warm-start', dest='warm_start', action='store_false',
                   help='Disable warm-up.')
    p.set_defaults(warm_start=True)

    # Outputs
    p.add_argument('--out-dir', type=str, required=True, help='Output directory.')
    p.add_argument('--save-model', type=str, default='statescope.pkl',
                   help='Saved model filename.')
    p.add_argument('--save-fractions', type=str, default='fractions3.csv',
                   help='Fractions CSV filename.')

    return p.parse_args()


# ------------------------- Utils ------------------------- #
def _auto_sep_for(path: str):
    if path.lower().endswith('.csv'):
        return ','
    if path.lower().endswith(('.tsv', '.txt')):
        return '\t'
    return None  # let pandas sniff


def _read_csv_matrix(path: str, kind: str) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        raise SystemExit(f"[error] {kind} file not found: {path}")
    sep = _auto_sep_for(path)
    try:
        df = pd.read_csv(path, index_col=0, sep=sep, engine="python")
    except Exception as e:
        raise SystemExit(f"[error] Could not parse {kind} file: {path}\n{e}")
    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise SystemExit(f"[error] {kind} appears empty: {path}")
    df = df.apply(pd.to_numeric, errors='coerce')
    before = df.shape
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if df.shape != before:
        print(f"[warn] Dropped all-NaN rows/cols from {kind}: {before} -> {df.shape}")
    if df.isna().values.any():
        print(f"[warn] {kind} contains NaNs → filling with zeros.")
        df = df.fillna(0.0)
    return df


def _load_expectation(exp_path: str, model: Statescope) -> pd.DataFrame | None:
    """
    Load an Expectation TSV, align rows to model.Samples
    and columns to model.Celltypes. Missing values remain NaN.
    """
    # read file (expecting tab-delimited, index = samples)
    df = pd.read_csv(exp_path, sep="\t", index_col=0)
    return df


def _align_signature_and_bulk(sig: pd.DataFrame, bulk: pd.DataFrame):
    common = sig.index.intersection(bulk.index)
    if common.empty:
        raise SystemExit("[error] No overlapping genes between Signature and Bulk.")
    sig2  = sig.loc[common]
    bulk2 = bulk.loc[common]
    drop = (sig2.abs().sum(axis=1) == 0) | (bulk2.abs().sum(axis=1) == 0)
    if drop.any():
        print(f"[warn] Dropping {int(drop.sum())} all-zero genes; keeping {int((~drop).sum())}")
        sig2  = sig2.loc[~drop]
        bulk2 = bulk2.loc[~drop]
    return sig2, bulk2


# ---------- Initializers (with forced CP10K on Bulk) ---------- #
def init_from_adata(adata_path: str, bulk_csv: str, celltype_key: str) -> Statescope:
    if not (bulk_csv and celltype_key):
        raise SystemExit("Using --adata requires --bulk-csv and --celltype-key.")
    import scanpy as sc
    from scipy.sparse import issparse

    adata = sc.read_h5ad(adata_path)
    if issparse(adata.X):
        adata.X = adata.X.toarray()

    Bulk = _read_csv_matrix(bulk_csv, "Bulk")


    return Initialize_Statescope(Bulk, Signature=adata, celltype_key=celltype_key)


def init_from_signature_csv(signature_csv: str, bulk_csv: str) -> Statescope:
    if not bulk_csv:
        raise SystemExit("Using --signature-csv requires --bulk-csv.")
    Sig  = _read_csv_matrix(signature_csv, "Signature")
    Bulk = _read_csv_matrix(bulk_csv, "Bulk")


    Sig, Bulk = _align_signature_and_bulk(Sig, Bulk)

    return Initialize_Statescope(Bulk, Signature=Sig)


# ------------------------- Main ------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Backend
    if args.backend == 'auto':
        backend = 'gpu' if torch.cuda.is_available() else 'cpu'
    elif args.backend == 'gpu' and not torch.cuda.is_available():
        print("[warn] GPU requested but not available → using CPU.")
        backend = 'cpu'
    else:
        backend = args.backend
    print(f"[info] Backend: {backend.upper()}")

    # Load or initialize model
    if args.model_path:
        print(f"[info] Loading model: {args.model_path}")
        model = Statescope.load(args.model_path, device=('cuda' if backend == 'gpu' else 'cpu'), blade_class=BLADE)

        if args.bulk_csv:
            Bulk = _read_csv_matrix(args.bulk_csv, "Bulk")

            if isinstance(getattr(model, "scExp", None), pd.DataFrame):
                sig_df = model.scExp
                sig_df, Bulk = _align_signature_and_bulk(sig_df, Bulk)
                print("[info] Re-initializing with existing signature + provided Bulk.")
                model = Initialize_Statescope(Bulk, Signature=sig_df)
            else:
                print("[info] Using provided Bulk with loaded model (signature baked in).")

    elif args.adata:
        print(f"[info] Initializing model from AnnData+Bulk: {args.adata} | {args.bulk_csv}")
        model = init_from_adata(args.adata, args.bulk_csv, args.celltype_key)

    elif args.signature_csv:
        print(f"[info] Initializing model from SignatureCSV+Bulk: {args.signature_csv} | {args.bulk_csv}")
        model = init_from_signature_csv(args.signature_csv, args.bulk_csv)

    else:
        # D) tumor-type route
        if not (args.tumor_type and args.ncelltypes and args.bulk_csv):
            raise SystemExit("Using --tumor-type requires --ncelltypes and --bulk-csv.")
        print(f"[info] Using built-in signature: TumorType={args.tumor_type}, Ncelltypes={args.ncelltypes}")

        Bulk = _read_csv_matrix(args.bulk_csv, "Bulk")
        
        col_sums = Bulk.sum(axis=0).astype(float)
        Bulk = Bulk.astype(float).div(col_sums, axis=1).mul(1e4)
        np.allclose(Bulk.sum(axis=0).to_numpy(), 1e4, rtol=1e-3, atol=1e-6)
        
        model = Initialize_Statescope(
            Bulk,
            Signature=None,
            TumorType=args.tumor_type,
            Ncelltypes=args.ncelltypes
        )

    # Expectation (optional)
    Expectation = None
    if args.expectation_csv:
        print(f"[info] Loading Expectation: {args.expectation_csv}")
        Expectation = _load_expectation(args.expectation_csv, model)

    # Shapes (best effort)
    try:
        print(f"[info] genes={len(model.Genes)}, celltypes={len(model.Celltypes)}, samples={len(model.Samples)}")
    except Exception:
        pass

    # Run deconvolution
    model.Deconvolution(
        Nrep=args.nrep,
        Njob=args.njobs,
        IterMax=args.iter_max,
        backend=backend,
        Expectation=Expectation,
        warm_start=args.warm_start,
        Alpha=1,
    )

    # Save outputs
    model_path = os.path.join(args.out_dir, args.save_model)
    frac_path  = os.path.join(args.out_dir, args.save_fractions)

    model.save(model_path, to_cpu=True)
    if not hasattr(model, "Fractions") or model.Fractions is None:
        raise SystemExit("[error] model.Fractions missing after Deconvolution.")
    model.Fractions.to_csv(frac_path)

    print(f"[done] Saved model: {model_path}")
    print(f"[done] Saved fractions: {frac_path}")


if __name__ == "__main__":
    main()
