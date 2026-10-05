# AIDO-BBA: modular black-box audit for cancer transcriptomic classification

AIDO-BBA is a reproducible computational audit workflow for separating aggregate discrimination from patient-level cross-model rank disagreement, resampling instability, held-out attribution, biological-process representation completeness, representation gaps, fuzzy explanatory allocation, and evidence-bounded audit follow-up triage.

## Scientific scope

The repository accompanies the manuscript **“A modular computational audit framework for high-dimensional transcriptomic classifiers: patient-level disagreement, explanatory completeness, and representation gaps.”** The primary implementation uses TCGA-BRCA stage classification. METABRIC, GSE96058, and TCGA-KIRC are replacement stress tests. Audit outputs are computational descriptors—not biological subtypes, mechanisms, diagnostic entities, clinical directives, or demonstrated clinical utility.

The 2026-10-04 repair line corrects several manuscript-facing definitions while preserving historical execution provenance. The repaired contract is documented in [`docs/SCIENTIFIC_REPAIR_CONTRACT_2026-10-04.md`](docs/SCIENTIFIC_REPAIR_CONTRACT_2026-10-04.md) and encoded in [`aido_bba/audit_contract.py`](aido_bba/audit_contract.py). In particular:

- primary cross-model disagreement uses calibration-insensitive rank separation `D_rank`, while raw probability separation is descriptive only;
- attribution-mass coverage is the gene-level mapped absolute attribution fraction;
- fuzzy memberships are independent per-axis magnitude memberships and are not constrained to sum to one;
- the historical mutually exclusive integrated patient taxonomy is retired from the repaired primary analysis;
- audit follow-up triage retains four source-verified overlapping computational flags, while the historical `model arbitration` action is superseded;
- repaired statistical-family counts exclude invalid groupings tied to the retired integrated taxonomy or raw-probability model-dependence tier.

## Repository map

```text
AIDO-BBA/
├── aido_bba/                  # configuration, schema validation, demo pipeline, repaired audit contract
├── configs/                   # example local-path configuration
├── notebooks/                 # sequential, reader-friendly notebooks
├── scripts/
│   ├── primary_brca/          # primary audit cells in execution order
│   ├── recommendation/        # specificity and constrained-null audits
│   └── external/              # METABRIC, GSE96058, and KIRC stress tests
├── demo/                      # deterministic self-contained execution demonstration
├── tests/                     # schema, deterministic, failure-state, and repair-contract tests
├── .github/workflows/         # continuous-integration workflow
├── docs/                      # workflow, data layout, output, and scientific repair contracts
└── legacy/                    # original development files retained for provenance
```

## Installation

```bash
conda env create -f environment.yml
conda activate aido-bba
```

or

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Local configuration

Copy the example file and edit only the local paths:

```bash
copy configs\config.example.json config.local.json
```

Linux/macOS:

```bash
cp configs/config.example.json config.local.json
```

`config.local.json` is excluded from Git because it contains machine-specific paths.

## Recommended execution order

1. `notebooks/01_primary_brca_audit.ipynb`
2. `scripts/recommendation/02_recommendation_specificity_audit.py`
3. `scripts/recommendation/03_recommendation_null_audit.py`
4. `scripts/external/04_metabric_dataset_replacement.py`
5. `scripts/external/05_gse96058_endpoint_replacement.py`
6. `scripts/external/06_kirc_cancer_type_replacement.py`

Run commands from the repository root so that the `aido_bba` package is importable:

```bash
python scripts/external/04_metabric_dataset_replacement.py
```

## Compact self-contained demonstration

A deterministic synthetic demonstration verifies the executable audit contract without downloading governed molecular datasets:

```bash
python demo/run_demo.py
```

The compact run validates input schemas, expression orientation, patient matching, endpoint parsing, repeated out-of-fold modelling, patient-level reliability outputs, simplified representation accounting, explicit failure logging, and output schemas. Generated files are written to `demo/demo_outputs/`.

The demonstration tests software execution and interface integrity. It does **not** reproduce the manuscript-scale SHAP analysis, biological results, or clinical claims.

## Automated tests and continuous integration

Install the development requirements and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover valid inputs, missing columns, duplicate or unmatched patients, invalid endpoint labels, non-numeric expression values, reversed expression orientation, deterministic output counts and schemas, explicit failure-state artifacts, and the repaired manuscript-facing audit definitions. GitHub Actions runs both the test suite and compact demonstration on pushes and pull requests.

## Data availability

The repository does **not** redistribute TCGA, METABRIC, GSE96058, GO/MSigDB, HGNC, or NCBI Gene files. Obtain data from their original repositories and arrange them as described in [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

## Reproducibility notes

- Feature selection is performed inside training folds.
- Patient-level model outputs are out-of-fold.
- TreeSHAP is computed on held-out samples and checked for exact additivity.
- Process reconstruction retains an explicit unmapped residual.
- Cross-model disagreement in the repaired manuscript is represented primarily by `D_rank`, not a hard raw-probability state.
- Fuzzy explanatory memberships are independent per-axis magnitudes; overlapping high memberships are valid by construction.
- Audit follow-up flags are overlapping computational outputs, not patient-management recommendations.
- External analyses retrain compatible models and are replacement stress tests, not frozen-model validation.
- Generated results, large matrices, and patient-level reports are ignored by Git by default.

## Citation and release status

See [`CITATION.cff`](CITATION.cff). Historical v1.0.1 remains provenance evidence. The repaired scientific line is integrated into `main`, metadata target v1.1.0, and release-finalization notes are maintained in [`RELEASE_NOTES_v1.1.0.md`](RELEASE_NOTES_v1.1.0.md). A v1.1.0 tag, GitHub Release, and archival DOI must be treated as final only after those release objects are explicitly created and verified.

## License

The current repository is provided for personal evaluation under the terms in [`LICENSE`](LICENSE). Do not describe the repository as open source unless and until an approved open-source license replaces the current terms.
