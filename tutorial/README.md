````markdown
# Statescope Deconvolution — Troubleshooting

This file lists the **knobs you can change** when things go wrong.

---

## Save and Load

### Save

Specify whether to save to CPU or GPU state.

```python
model.save("/path/to/model.pkl", to_cpu=True)   # save with CPU tensors
model.save("/path/to/model.pkl", to_cpu=False)  # save with GPU tensors
```

### Load

If the model was saved in a previous version, you may need to pass the `blade_class`.

```python
from BLADE_Deconvolution.BLADE import BLADE

model = Statescope.load(
    "/path/to/model.pkl",
    blade_class=BLADE
)
```
---

## Slow convergence

```python
warm_start=True | False        # toggle Adam warm-up before L-BFGS
IterMax=2000                   # increase iterations
lbfgs_params["lr"] = 0.01–0.1  # adjust learning rate
adam_params["lr"] = 1e-3–1e-4
Nrep=more                      # more restarts
````

---

## Non-finite ELBO

```python
lbfgs_params["lr"] = lower
adam_params["lr"] = lower
adam_params["grad_clip"] = 10000.0  # clamp exploding grads
Expectation[...] = NaN or (0,1)     # never exact 0 or 1
```

---

## CUDA OOM

Set an **expected VRAM per rep (GiB)** in `lbfgs_params`:

```python
lbfgs_params = {
    "lr": 0.05,
    "steps": 12,
    "max_iter": 12,
    "history_size": 100,
    "est_vram_gb": 1.25,   # expected GiB per rep
}
```

* This value is compared to free GPU memory and jobs are capped so total use ≤ ~85%.
* Use `1.25` for ~1000 samples × 1500 genes × 15 cell types.
* Scale up for more genes/celltypes.

Extra:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Backend

```python
backend="cpu" | "gpu" | "auto"
```

---

## Priors

```python
Expectation = pd.DataFrame(np.nan, index=model.Samples, columns=model.Celltypes)
Expectation["T cell"] = [0.3, 0.25, 0.4]  # fill known priors
```

---

## Optimizer dicts

```python
adam_params = {
  "lr": 0.001,
  "steps": 200,
  "betas": [0.9, 0.98],
  "grad_clip": 10000.0
}

lbfgs_params = {
  "lr": 0.05,
  "steps": 12,
  "max_iter": 12,
  "history_size": 100,
  "est_vram_gb": 1.25   # <— VRAM guard
}
```

Pass them:

```python
model.Deconvolution(adam_params=adam_params, lbfgs_params=lbfgs_params)
```

---

## Check VRAM

```python
used = torch.cuda.max_memory_allocated()/(1024**3)
print(f"peak VRAM ≈ {used:.2f} GiB")
```

---


