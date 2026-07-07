# Compact self-contained demonstration

This deterministic synthetic demonstration verifies the AIDO-BBA software contract without requiring TCGA, METABRIC, GSE96058, KIRC, MSigDB, HGNC, or NCBI files.

From the repository root, run:

```bash
python demo/run_demo.py
```

The demonstration checks input schemas, patient matching, endpoint parsing, repeated out-of-fold modelling, patient-level reliability outputs, simplified representation accounting, explicit failure logging, and output schemas.

The synthetic demonstration is an execution and interface test. It does **not** reproduce the manuscript-scale SHAP analysis, biological results, or clinical claims.
