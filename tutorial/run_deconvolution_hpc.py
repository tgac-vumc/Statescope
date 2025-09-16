#!/usr/bin/env python3
"""
Statescope - Minimal Deconvolution Runner (SLURM/HPC-friendly)

Basic functionality only:
- Load an existing Statescope model, OR initialize from:
    (A) AnnData + Bulk  (requires --adata, --bulk-csv, --celltype-key), or
    (B) Signature matrix as CSV + Bulk (requires --signature-csv, --bulk-csv; NO --celltype-key)
- Optionally load an Expectation CSV to guide specific cell types.
- Run deconvolution on CPU or GPU.
- Save model and Fractions.

No optimizer defaults here, no runtime timing, no VRAM reporting.

Usage examples:

  # Existing model
  python run_deconvolution.py \
    --model-path /path/to/Test_model.pkl \
    --out-dir /path/to/results

  # From AnnData + Bulk (needs celltype key in .obs)
  python run_deconvolution.py \
    --adata /path/to/signature.h5ad \
    --bulk-csv /path/to/bulk_counts.csv \
    --celltype-key cell_type_coarse \
    --out-dir /path/to/results

  # From Signature CSV + Bulk (NO celltype key needed)
  # CSV shape: genes x celltypes (rows=genes, columns=cell types)
  python run_deconvolution.py \
    --signature-csv /path/to/signature_matrix.csv \
    --bulk-csv /path/to/bulk_counts.csv \
    --out-dir /path/to/results

  # Optional Expectation CSV (rows=samples, cols=subset of cell types)
  python run_deconvolution.py \
    --model-path /path/to/Test_model.pkl \
    --expectation-csv /path/to/Expectation.csv \
    --backend gpu \
    --nrep 4 --njobs 4 \
    --iter-max 1000 \
    --out-dir /path/to/results
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch


THIS_DIR = os.path.dirname(__file__)
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Statescope.Statescope import Initialize_Statescope, Statescope
from BLADE_Deconvolution.BLADEpro import BLADE

def parse_args():
    p = argparse.ArgumentParser(description="Run Statescope deconvolution (basic).")

    # Inputs (choose ONE: model OR adata OR signature CSV)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model-path', type=str, help='Path to an existing saved Statescope model (.pkl).')
    g.add_argument('--adata', type=str, help='Path to signature AnnData (.h5ad). Use with --bulk-csv and --celltype-key.')
    g.add_argument('--signature-csv', type=str, help='Path to signature matrix CSV (genes x celltypes). Use with --bulk-csv.')

    p.add_argument('--bulk-csv', type=str, help='Bulk expression as CSV (genes x samples). Required with --adata or --signature-csv.')
    p.add_argument('--celltype-key', type=str, help='AnnData .obs column with cell type labels (only for --adata).')

    # Optional Expectation
    p.add_argument('--expectation-csv', type=str, default=None,
                   help='Optional CSV of expected fractions; rows=samples, cols=subset of celltypes.')

    # Backend/parallel basics
    p.add_argument('--backend', choices=['auto','cpu','gpu'], default='auto', help='Device backend selection.')
    p.add_argument('--nrep', type=int, default=4, help='Number of random restarts.')
    p.add_argument('--njobs', type=int, default=4, help='Parallel jobs (processes).')
    p.add_argument('--iter-max', type=int, default=1000, help='Max outer iterations.')

    # Outputs
    p.add_argument('--out-dir', type=str, required=True, help='Output directory.')
    p.add_argument('--save-model', type=str, default='statescope_model2.pkl', help='Filename for saved model.')
    p.add_argument('--save-fractions', type=str, default='fractions2.csv', help='Filename for fractions CSV.')

    return p.parse_args()


def load_expectation(exp_csv: str, model: Statescope) -> pd.DataFrame:
    df = pd.read_csv(exp_csv, index_col=0)
    model_cols = list(model.Celltypes)
    lower_map = {c.lower(): c for c in model_cols}

    out = pd.DataFrame(np.nan, index=model.Samples, columns=model_cols, dtype=float)
    for c in df.columns:
        key = c.lower()
        if key not in lower_map:
            print(f"[warn] Expectation column '{c}' not found in model cell types; skipping.")
            continue
        tgt = lower_map[key]
        col = (df[c].reindex(model.Samples).astype(float)).clip(0.01, 0.99)
        out[tgt] = col
    return out


def init_from_adata(adata_path: str, bulk_csv: str, celltype_key: str) -> Statescope:
    if not (bulk_csv and celltype_key):
        raise SystemExit("Using --adata requires --bulk-csv and --celltype-key.")
    import scanpy as sc
    from scipy.sparse import issparse
    adata = sc.read_h5ad(adata_path)
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    Bulk = pd.read_csv(bulk_csv, index_col=0)
    return Initialize_Statescope(Bulk, Signature=adata, celltype_key=celltype_key)


def init_from_signature_csv(signature_csv: str, bulk_csv: str) -> Statescope:
    if not bulk_csv:
        raise SystemExit("Using --signature-csv requires --bulk-csv.")
    # Expect shape: genes x celltypes
    Sig = pd.read_csv(signature_csv, index_col=0)
    Bulk = pd.read_csv(bulk_csv, index_col=0)

    # Align by common genes (rows)
    common = Sig.index.intersection(Bulk.index)
    if common.empty:
        raise SystemExit("No overlapping genes between signature CSV and bulk CSV.")
    Sig = Sig.loc[common]
    Bulk = Bulk.loc[common]

    # Initialize without celltype_key (not needed for DataFrame signature)
    return Initialize_Statescope(Bulk, Signature=Sig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load or initialize model
    if args.model_path:
        print(f"[info] Loading model: {args.model_path}")
        model = Statescope.load(args.model_path, blade_class=BLADE)
    elif args.adata:
        print(f"[info] Initializing model from AnnData+Bulk: {args.adata} | {args.bulk_csv}")
        model = init_from_adata(args.adata, args.bulk_csv, args.celltype_key)
    else:
        print(f"[info] Initializing model from SignatureCSV+Bulk: {args.signature_csv} | {args.bulk_csv}")
        model = init_from_signature_csv(args.signature_csv, args.bulk_csv)

    # Expectation (optional)
    Expectation = None
    if args.expectation_csv:
        print(f"[info] Loading Expectation: {args.expectation_csv}")
        Expectation = load_expectation(args.expectation_csv, model)

    # Backend
    if args.backend == 'auto':
        backend = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        backend = args.backend
    print(f"[info] Backend: {backend.upper()}")

    # Run deconvolution (basic call)
    model.Deconvolution(
        Nrep=args.nrep,
        Njob=args.njobs,  
        IterMax=args.iter_max,
        backend=backend,
        Expectation=Expectation
    )

    # Save outputs
    model_path = os.path.join(args.out_dir, args.save_model)
    frac_path  = os.path.join(args.out_dir, args.save_fractions)

    model.save(model_path, to_cpu=True)
    model.Fractions.to_csv(frac_path)

    print(f"[done] Saved model: {model_path}")
    print(f"[done] Saved fractions: {frac_path}")


if __name__ == "__main__":
    main()
